/* C++ NIF wrapper around llama.cpp's common_chat_parse autoparser
 * (vendored at apps/erllama/c_src/llama.cpp/common/chat.cpp).
 *
 * Three entry points exposed to Erlang via the existing
 * erllama_nif .so:
 *
 *   nif_chat_templates_init(ModelRef, TemplateOverride)
 *     -> {ok, ChatTemplatesRef} | {error, Reason}
 *
 *   nif_chat_templates_apply(ChatTemplatesRef, InputsMap)
 *     -> {ok, ChatParamsRef, RenderMap} | {error, Reason}
 *     RenderMap carries the rendered prompt plus the constraint set
 *     the template pass synthesized (grammar, lazy triggers,
 *     additional stops, generation prompt, thinking tags).
 *
 *   nif_chat_parse(ChatParamsRef, Input, IsPartial)
 *     -> {ok, ParsedMsg} | {error, Reason}
 *
 * All three run on ERL_NIF_DIRTY_JOB_CPU_BOUND. C++ exceptions are
 * caught and converted to `{error, {chat_parse_failed, Reason}}'.
 *
 * Resources have C++ destructors (unique_ptr resets,
 * placement-deletes) registered through enif_open_resource_type.
 */

#include "erllama_chat_nif.h"
#include "erllama_nif_util.h"
#include "erllama_resources.h"
#include "chat.h"
#include "chat-peg-parser.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>

/* ============================================================== */
/* Resources                                                      */
/* ============================================================== */

static ErlNifResourceType *CHAT_TEMPLATES_RT = nullptr;
static ErlNifResourceType *CHAT_PARAMS_RT    = nullptr;

namespace {

struct chat_templates_holder {
    common_chat_templates_ptr ptr;
};

struct chat_params_holder {
    common_chat_params params;
};

void chat_templates_dtor(ErlNifEnv *, void *obj) {
    auto *h = static_cast<chat_templates_holder *>(obj);
    /* unique_ptr destructor releases the templates via the
     * registered common_chat_templates_free deleter. */
    h->~chat_templates_holder();
}

void chat_params_dtor(ErlNifEnv *, void *obj) {
    auto *h = static_cast<chat_params_holder *>(obj);
    /* common_chat_params holds the PEG arena + the synthesised
     * grammar as value-typed fields; their destructors run via the
     * struct destructor we explicitly invoke here. */
    h->~chat_params_holder();
}

ERL_NIF_TERM mk_atom(ErlNifEnv *env, const char *name) {
    return enif_make_atom(env, name);
}

ERL_NIF_TERM mk_error(ErlNifEnv *env, const char *reason) {
    return enif_make_tuple2(env, mk_atom(env, "error"), mk_atom(env, reason));
}

ERL_NIF_TERM mk_error_str(ErlNifEnv *env, const std::string &reason) {
    ERL_NIF_TERM bin;
    if (!erllama_bin_from(env, reason.data(), reason.size(), &bin)) {
        return mk_error(env, "alloc_failed");
    }
    return enif_make_tuple2(
        env,
        mk_atom(env, "error"),
        enif_make_tuple2(env, mk_atom(env, "chat_parse_failed"), bin));
}

bool term_to_string(ErlNifEnv *env, ERL_NIF_TERM t, std::string &out) {
    ErlNifBinary bin;
    if (!enif_inspect_iolist_as_binary(env, t, &bin)) {
        return false;
    }
    out.assign(reinterpret_cast<const char *>(bin.data), bin.size);
    return true;
}

bool term_is_undefined(ErlNifEnv *env, ERL_NIF_TERM t) {
    char buf[16];
    if (enif_get_atom(env, t, buf, sizeof(buf), ERL_NIF_LATIN1) == 0) {
        return false;
    }
    return std::string(buf) == "undefined";
}

} /* anonymous namespace */

/* ============================================================== */
/* Load                                                           */
/* ============================================================== */

extern "C" int chat_nif_load(ErlNifEnv *env) {
    ErlNifResourceFlags tried = ERL_NIF_RT_CREATE;

    CHAT_TEMPLATES_RT = enif_open_resource_type(
        env, nullptr, "erllama_chat_templates",
        chat_templates_dtor, tried, nullptr);
    if (!CHAT_TEMPLATES_RT) {
        return -1;
    }

    CHAT_PARAMS_RT = enif_open_resource_type(
        env, nullptr, "erllama_chat_params",
        chat_params_dtor, tried, nullptr);
    if (!CHAT_PARAMS_RT) {
        return -1;
    }

    return 0;
}

/* ============================================================== */
/* nif_chat_templates_init                                        */
/* ============================================================== */

extern "C" ERL_NIF_TERM nif_chat_templates_init(
    ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 2) {
        return enif_make_badarg(env);
    }

    erllama_model_t *m = nullptr;
    if (!enif_get_resource(env, argv[0], MODEL_RT, (void **) &m)) {
        return mk_error(env, "invalid_model");
    }
    if (m->model == nullptr) {
        return mk_error(env, "model_released");
    }

    std::string override_template;
    bool have_override = !term_is_undefined(env, argv[1]);
    if (have_override) {
        if (!term_to_string(env, argv[1], override_template)) {
            return enif_make_badarg(env);
        }
    }

    try {
        common_chat_templates_ptr ptr = common_chat_templates_init(
            m->model,
            have_override ? override_template : std::string(),
            /* bos_token_override = */ std::string(),
            /* eos_token_override = */ std::string());
        if (!ptr) {
            return mk_error(env, "templates_init_failed");
        }

        void *res = enif_alloc_resource(
            CHAT_TEMPLATES_RT, sizeof(chat_templates_holder));
        if (!res) {
            return mk_error(env, "alloc_failed");
        }
        new (res) chat_templates_holder{std::move(ptr)};

        ERL_NIF_TERM term = enif_make_resource(env, res);
        enif_release_resource(res);
        return enif_make_tuple2(env, mk_atom(env, "ok"), term);
    } catch (const std::exception &e) {
        return mk_error_str(env, e.what());
    } catch (...) {
        return mk_error(env, "unknown_exception");
    }
}

/* ============================================================== */
/* Helpers for the apply / parse paths                            */
/* ============================================================== */

namespace {

/* Pull an iolist/binary key from an Erlang map and convert to a
 * std::string. Returns true on success; false if the key is
 * absent or has the wrong type. */
bool map_get_string(
    ErlNifEnv *env, ERL_NIF_TERM map, const char *key, std::string &out) {
    ERL_NIF_TERM kterm = enif_make_atom(env, key);
    ERL_NIF_TERM v;
    if (!enif_get_map_value(env, map, kterm, &v)) {
        return false;
    }
    return term_to_string(env, v, out);
}

/* Atom -> enum tables consumed through the shared get_map_atom_enum
 * getter (erllama_nif_util.h), mirroring the C-side option maps. */
const erllama_atom_enum_pair_t TOOL_CHOICE_TABLE[] = {
    {"auto",     COMMON_CHAT_TOOL_CHOICE_AUTO},
    {"required", COMMON_CHAT_TOOL_CHOICE_REQUIRED},
    {"none",     COMMON_CHAT_TOOL_CHOICE_NONE},
};

const erllama_atom_enum_pair_t REASONING_FORMAT_TABLE[] = {
    {"none",     COMMON_REASONING_FORMAT_NONE},
    {"deepseek", COMMON_REASONING_FORMAT_DEEPSEEK},
};

const erllama_atom_enum_pair_t CONTINUATION_TABLE[] = {
    {"none",      COMMON_CHAT_CONTINUATION_NONE},
    {"auto",      COMMON_CHAT_CONTINUATION_AUTO},
    {"content",   COMMON_CHAT_CONTINUATION_CONTENT},
    {"reasoning", COMMON_CHAT_CONTINUATION_REASONING},
};

/* Map atom keys `auto' | `required' | `none' -> common_chat_tool_choice.
 * Missing key -> AUTO; anything unrecognised is rejected. */
bool map_tool_choice(ErlNifEnv *env, ERL_NIF_TERM map,
                     common_chat_tool_choice &out) {
    int v = 0;
    int rc = get_map_atom_enum(
        env, map, "tool_choice", TOOL_CHOICE_TABLE,
        sizeof(TOOL_CHOICE_TABLE) / sizeof(TOOL_CHOICE_TABLE[0]), &v);
    if (rc < 0) {
        return false;
    }
    out = rc > 0 ? static_cast<common_chat_tool_choice>(v)
                 : COMMON_CHAT_TOOL_CHOICE_AUTO;
    return true;
}

/* Boolean atom key. Missing -> `deflt'; non-boolean values rejected.
 * Presence is checked first so the shared get_map_bool (which folds
 * "absent" and "not a boolean" into one return) keeps the rejection
 * behavior here. */
bool map_get_bool(ErlNifEnv *env, ERL_NIF_TERM map, const char *key,
                  bool deflt, bool &out) {
    ERL_NIF_TERM v;
    if (!enif_get_map_value(env, map, enif_make_atom(env, key), &v)) {
        out = deflt;
        return true;
    }
    int b = 0;
    if (!get_map_bool(env, map, key, &b)) {
        return false;
    }
    out = b != 0;
    return true;
}

/* `reasoning_format' atom -> common_reasoning_format. Missing key
 * keeps the struct default (NONE); the Erlang side always sets it. */
bool map_reasoning_format(ErlNifEnv *env, ERL_NIF_TERM map,
                          common_reasoning_format &out) {
    int v = 0;
    int rc = get_map_atom_enum(
        env, map, "reasoning_format", REASONING_FORMAT_TABLE,
        sizeof(REASONING_FORMAT_TABLE) / sizeof(REASONING_FORMAT_TABLE[0]),
        &v);
    if (rc < 0) {
        return false;
    }
    if (rc > 0) {
        out = static_cast<common_reasoning_format>(v);
    }
    return true;
}

/* `continue_final_message' atom -> common_chat_continuation.
 * Missing key keeps the struct default (NONE). */
bool map_continuation(ErlNifEnv *env, ERL_NIF_TERM map,
                      common_chat_continuation &out) {
    int v = 0;
    int rc = get_map_atom_enum(
        env, map, "continue_final_message", CONTINUATION_TABLE,
        sizeof(CONTINUATION_TABLE) / sizeof(CONTINUATION_TABLE[0]), &v);
    if (rc < 0) {
        return false;
    }
    if (rc > 0) {
        out = static_cast<common_chat_continuation>(v);
    }
    return true;
}

ERL_NIF_TERM mk_string_bin(ErlNifEnv *env, const std::string &s) {
    ERL_NIF_TERM bin;
    if (!erllama_bin_from(env, s.data(), s.size(), &bin)) {
        return enif_make_atom(env, "undefined");
    }
    return bin;
}

/* Convert a common_chat_msg to the Erlang map shape:
 *   #{
 *     role := binary(),
 *     content := binary(),
 *     reasoning_content := binary() | undefined,
 *     tool_calls := [#{name, arguments_json, id}]
 *   }
 * arguments stay as a JSON binary; the Erlang facade decodes via
 * json:decode/1 so the NIF layer carries no JSON-decode logic. */
ERL_NIF_TERM marshal_msg(ErlNifEnv *env, const common_chat_msg &msg) {
    ERL_NIF_TERM keys[4] = {
        enif_make_atom(env, "role"),
        enif_make_atom(env, "content"),
        enif_make_atom(env, "reasoning_content"),
        enif_make_atom(env, "tool_calls"),
    };

    ERL_NIF_TERM reasoning =
        msg.reasoning_content.empty()
            ? enif_make_atom(env, "undefined")
            : mk_string_bin(env, msg.reasoning_content);

    std::vector<ERL_NIF_TERM> calls;
    calls.reserve(msg.tool_calls.size());
    for (const auto &c : msg.tool_calls) {
        ERL_NIF_TERM ckeys[3] = {
            enif_make_atom(env, "name"),
            enif_make_atom(env, "arguments_json"),
            enif_make_atom(env, "id"),
        };
        ERL_NIF_TERM idterm = c.id.empty()
                                  ? enif_make_atom(env, "undefined")
                                  : mk_string_bin(env, c.id);
        ERL_NIF_TERM cvals[3] = {
            mk_string_bin(env, c.name),
            mk_string_bin(env, c.arguments),
            idterm,
        };
        ERL_NIF_TERM cmap;
        if (!enif_make_map_from_arrays(env, ckeys, cvals, 3, &cmap)) {
            cmap = enif_make_atom(env, "undefined");
        }
        calls.push_back(cmap);
    }
    ERL_NIF_TERM calls_list =
        enif_make_list_from_array(env, calls.data(), calls.size());

    ERL_NIF_TERM vals[4] = {
        mk_string_bin(env, msg.role),
        mk_string_bin(env, msg.content),
        reasoning,
        calls_list,
    };

    ERL_NIF_TERM out;
    if (!enif_make_map_from_arrays(env, keys, vals, 4, &out)) {
        return mk_error(env, "marshal_failed");
    }
    return out;
}

} /* anonymous namespace */

/* Shared inputs-map -> common_chat_templates_inputs conversion used by
 * nif_chat_templates_apply.
 * Returns false (caller should return mk_error) on missing required
 * keys; true on success. */
static bool build_chat_inputs_from_map(ErlNifEnv *env, ERL_NIF_TERM map,
                                       common_chat_templates_inputs &inputs,
                                       const char **err_out) {
    std::string messages_json;
    if (!map_get_string(env, map, "messages", messages_json)) {
        *err_out = "missing_messages";
        return false;
    }
    std::string tools_json;
    bool have_tools = map_get_string(env, map, "tools", tools_json);
    inputs.use_jinja = true;
    inputs.messages =
        common_chat_msgs_parse_oaicompat(common_json::parse(messages_json));
    if (have_tools && !tools_json.empty()) {
        inputs.tools =
            common_chat_tools_parse_oaicompat(common_json::parse(tools_json));
    }
    if (!map_tool_choice(env, map, inputs.tool_choice)) {
        *err_out = "invalid_tool_choice";
        return false;
    }
    if (!map_get_bool(env, map, "parallel_tool_calls", false,
                      inputs.parallel_tool_calls)) {
        *err_out = "invalid_parallel_tool_calls";
        return false;
    }
    /* Optional response-format schema (raw JSON text). */
    std::string json_schema;
    if (map_get_string(env, map, "json_schema", json_schema) &&
        !json_schema.empty()) {
        inputs.json_schema = json_schema;
    }
    if (!map_get_bool(env, map, "enable_thinking", inputs.enable_thinking,
                      inputs.enable_thinking)) {
        *err_out = "invalid_enable_thinking";
        return false;
    }
    if (!map_reasoning_format(env, map, inputs.reasoning_format)) {
        *err_out = "invalid_reasoning_format";
        return false;
    }
    if (!map_continuation(env, map, inputs.continue_final_message)) {
        *err_out = "invalid_continue_final_message";
        return false;
    }
    return true;
}

/* ============================================================== */
/* Render map: the constraint set synthesized by the template pass */
/* ============================================================== */

namespace {

ERL_NIF_TERM mk_bool(ErlNifEnv *env, bool b) {
    return enif_make_atom(env, b ? "true" : "false");
}

ERL_NIF_TERM mk_bin_list(ErlNifEnv *env, const std::vector<std::string> &v) {
    std::vector<ERL_NIF_TERM> terms;
    terms.reserve(v.size());
    for (const auto &s : v) {
        terms.push_back(mk_string_bin(env, s));
    }
    return enif_make_list_from_array(env, terms.data(), terms.size());
}

/* Convert common_grammar_trigger entries to the shapes
 * llama_sampler_init_grammar_lazy_patterns takes, mirroring
 * common_sampler_init (common/sampling.cpp:220-256): WORD is
 * regex-escaped, PATTERN passes verbatim, PATTERN_FULL is anchored
 * ^...$, TOKEN goes to the token list. */
void convert_triggers(const std::vector<common_grammar_trigger> &triggers,
                      std::vector<std::string> &patterns,
                      std::vector<llama_token> &tokens) {
    for (const auto &t : triggers) {
        switch (t.type) {
            case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                patterns.push_back(regex_escape(t.value));
                break;
            case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                patterns.push_back(t.value);
                break;
            case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL: {
                std::string anchored = "^$";
                if (!t.value.empty()) {
                    anchored = std::string(t.value.front() == '^' ? "" : "^") +
                               t.value +
                               (t.value.back() == '$' ? "" : "$");
                }
                patterns.push_back(anchored);
                break;
            }
            case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                tokens.push_back(t.token);
                break;
        }
    }
}

/* #{prompt, format, grammar, grammar_lazy, trigger_patterns,
 *   trigger_tokens, additional_stops, generation_prompt,
 *   supports_thinking, thinking_start_tag, thinking_end_tags} */
ERL_NIF_TERM render_map(ErlNifEnv *env, const common_chat_params &params) {
    std::vector<std::string> patterns;
    std::vector<llama_token> tokens;
    convert_triggers(params.grammar_triggers, patterns, tokens);

    std::vector<ERL_NIF_TERM> tok_terms;
    tok_terms.reserve(tokens.size());
    for (llama_token t : tokens) {
        tok_terms.push_back(enif_make_int(env, t));
    }

    ERL_NIF_TERM keys[11] = {
        enif_make_atom(env, "prompt"),
        enif_make_atom(env, "format"),
        enif_make_atom(env, "grammar"),
        enif_make_atom(env, "grammar_lazy"),
        enif_make_atom(env, "trigger_patterns"),
        enif_make_atom(env, "trigger_tokens"),
        enif_make_atom(env, "additional_stops"),
        enif_make_atom(env, "generation_prompt"),
        enif_make_atom(env, "supports_thinking"),
        enif_make_atom(env, "thinking_start_tag"),
        enif_make_atom(env, "thinking_end_tags"),
    };
    ERL_NIF_TERM vals[11] = {
        mk_string_bin(env, params.prompt),
        mk_string_bin(env, common_chat_format_name(params.format)),
        mk_string_bin(env, params.grammar),
        mk_bool(env, params.grammar_lazy),
        mk_bin_list(env, patterns),
        enif_make_list_from_array(env, tok_terms.data(), tok_terms.size()),
        mk_bin_list(env, params.additional_stops),
        mk_string_bin(env, params.generation_prompt),
        mk_bool(env, params.supports_thinking),
        mk_string_bin(env, params.thinking_start_tag),
        mk_bin_list(env, params.thinking_end_tags),
    };
    ERL_NIF_TERM out;
    if (!enif_make_map_from_arrays(env, keys, vals, 11, &out)) {
        return mk_error(env, "marshal_failed");
    }
    return out;
}

} /* anonymous namespace */

/* ============================================================== */
/* nif_chat_templates_apply                                       */
/* ============================================================== */

extern "C" ERL_NIF_TERM nif_chat_templates_apply(
    ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 2) {
        return enif_make_badarg(env);
    }
    chat_templates_holder *th = nullptr;
    if (!enif_get_resource(env, argv[0], CHAT_TEMPLATES_RT, (void **) &th)) {
        return mk_error(env, "invalid_templates");
    }
    if (!enif_is_map(env, argv[1])) {
        return enif_make_badarg(env);
    }

    try {
        common_chat_templates_inputs inputs;
        const char *err = nullptr;
        if (!build_chat_inputs_from_map(env, argv[1], inputs, &err)) {
            return mk_error(env, err);
        }
        common_chat_params params =
            common_chat_templates_apply(th->ptr.get(), inputs);

        void *res = enif_alloc_resource(
            CHAT_PARAMS_RT, sizeof(chat_params_holder));
        if (!res) {
            return mk_error(env, "alloc_failed");
        }
        new (res) chat_params_holder{std::move(params)};

        auto *holder = static_cast<chat_params_holder *>(res);
        ERL_NIF_TERM ref = enif_make_resource(env, res);
        ERL_NIF_TERM render = render_map(env, holder->params);
        enif_release_resource(res);
        return enif_make_tuple3(env, mk_atom(env, "ok"), ref, render);
    } catch (const std::exception &e) {
        return mk_error_str(env, e.what());
    } catch (...) {
        return mk_error(env, "unknown_exception");
    }
}

/* ============================================================== */
/* nif_chat_parse                                                 */
/* ============================================================== */

extern "C" ERL_NIF_TERM nif_chat_parse(
    ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 3) {
        return enif_make_badarg(env);
    }
    chat_params_holder *ph = nullptr;
    if (!enif_get_resource(env, argv[0], CHAT_PARAMS_RT, (void **) &ph)) {
        return mk_error(env, "invalid_params");
    }
    ErlNifBinary input;
    if (!enif_inspect_iolist_as_binary(env, argv[1], &input)) {
        return enif_make_badarg(env);
    }
    char ispart[8];
    if (enif_get_atom(env, argv[2], ispart, sizeof(ispart), ERL_NIF_LATIN1) == 0) {
        return enif_make_badarg(env);
    }
    bool is_partial = std::string(ispart) == "true";

    try {
        common_chat_parser_params parser_params(ph->params);
        parser_params.parser.load(ph->params.parser);

        std::string input_str(
            reinterpret_cast<const char *>(input.data), input.size);
        common_chat_msg msg =
            common_chat_parse(input_str, is_partial, parser_params);

        return enif_make_tuple2(env, mk_atom(env, "ok"), marshal_msg(env, msg));
    } catch (const std::exception &e) {
        return mk_error_str(env, e.what());
    } catch (...) {
        return mk_error(env, "unknown_exception");
    }
}
