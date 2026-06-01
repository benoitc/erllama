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

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>

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
/* nif_chat_templates_apply                                       */
/* nif_chat_parse                                                 */
/* ============================================================== */
/* Phase 3.B ships the build wiring + the templates_init path as a */
/* dormant capability. The Inputs->common_chat_templates_inputs    */
/* translation and the parse->Erlang-term marshalling land in the  */
/* next iteration (the cache + facade modules in Erlang can call   */
/* templates_init today; apply / parse return {error,              */
/* not_implemented} so callers know not to depend on them yet).    */

extern "C" ERL_NIF_TERM nif_chat_templates_apply(
    ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    (void) argv;
    return mk_error(env, "not_implemented");
}

extern "C" ERL_NIF_TERM nif_chat_parse(
    ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    (void) argv;
    return mk_error(env, "not_implemented");
}
