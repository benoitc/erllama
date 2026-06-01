/* C++ NIF wrapper around llama.cpp's common_chat_parse autoparser
 * (vendored at apps/barrel_inference/c_src/llama.cpp/common/chat.cpp).
 *
 * Three entry points exposed to Erlang via the existing
 * barrel_inference_nif .so:
 *
 *   nif_chat_templates_init(ModelRef, TemplateOverride)
 *     -> {ok, ChatTemplatesRef} | {error, Reason}
 *
 *   nif_chat_templates_apply(ChatTemplatesRef, InputsJSON)
 *     -> {ok, ChatParamsRef, PromptBin} | {error, Reason}
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

#include "barrel_inference_chat_nif.h"
#include "barrel_inference_resources.h"
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
    unsigned char *buf =
        enif_make_new_binary(env, reason.size(), &bin);
    if (!buf) {
        return mk_error(env, "alloc_failed");
    }
    /* enif_make_new_binary returns a raw binary buffer; null
     * termination is not required (this is an Erlang binary,
     * not a C string). std::copy avoids clang-tidy's
     * bugprone-not-null-terminated-result false positive on
     * memcpy. */
    std::copy(reason.begin(), reason.end(), buf);
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
        env, nullptr, "barrel_inference_chat_templates",
        chat_templates_dtor, tried, nullptr);
    if (!CHAT_TEMPLATES_RT) {
        return -1;
    }

    CHAT_PARAMS_RT = enif_open_resource_type(
        env, nullptr, "barrel_inference_chat_params",
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

    barrel_inference_model_t *m = nullptr;
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

/* Map atom keys `auto' | `required' | `none' -> common_chat_tool_choice.
 * Defaults to AUTO when the key is missing or the value is unrecognised. */
common_chat_tool_choice map_tool_choice(ErlNifEnv *env, ERL_NIF_TERM map) {
    ERL_NIF_TERM kterm = enif_make_atom(env, "tool_choice");
    ERL_NIF_TERM v;
    if (!enif_get_map_value(env, map, kterm, &v)) {
        return COMMON_CHAT_TOOL_CHOICE_AUTO;
    }
    char buf[16];
    if (enif_get_atom(env, v, buf, sizeof(buf), ERL_NIF_LATIN1) == 0) {
        return COMMON_CHAT_TOOL_CHOICE_AUTO;
    }
    std::string s(buf);
    if (s == "required") {
        return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    }
    if (s == "none") {
        return COMMON_CHAT_TOOL_CHOICE_NONE;
    }
    return COMMON_CHAT_TOOL_CHOICE_AUTO;
}

ERL_NIF_TERM mk_string_bin(ErlNifEnv *env, const std::string &s) {
    ERL_NIF_TERM bin;
    unsigned char *buf = enif_make_new_binary(env, s.size(), &bin);
    if (!buf) {
        return enif_make_atom(env, "undefined");
    }
    std::copy(s.begin(), s.end(), buf);
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

    /* Required keys: `messages' (JSON array binary). Optional:
     * `tools', `tool_choice', `add_generation_prompt'. */
    std::string messages_json;
    if (!map_get_string(env, argv[1], "messages", messages_json)) {
        return mk_error(env, "missing_messages");
    }
    std::string tools_json;
    bool have_tools = map_get_string(env, argv[1], "tools", tools_json);

    try {
        common_chat_templates_inputs inputs;
        inputs.use_jinja = true;
        inputs.messages =
            common_chat_msgs_parse_oaicompat(nlohmann::json::parse(messages_json));
        if (have_tools && !tools_json.empty()) {
            inputs.tools =
                common_chat_tools_parse_oaicompat(nlohmann::json::parse(tools_json));
        }
        inputs.tool_choice = map_tool_choice(env, argv[1]);

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
        ERL_NIF_TERM prompt = mk_string_bin(env, holder->params.prompt);
        enif_release_resource(res);
        return enif_make_tuple3(env, mk_atom(env, "ok"), ref, prompt);
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
