/* Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
 * See the LICENSE file at the project root. */
/* Legacy chat templating: render a normalised chat request
 * through the GGUF template and tokenize (pre-autoparser path). */

#include "erllama_nif_int.h"
#include "erllama_safe.h"

#include <math.h>
#include <string.h>

/* =========================================================================
 * Chat templating
 * =========================================================================
 *
 * nif_apply_chat_template renders a normalised chat request through
 * the model's chat template (read from GGUF metadata) and tokenises
 * the result. The Request map carries:
 *
 *   #{ messages := [#{role := binary(), content := binary()}]
 *    , system   => binary() | undefined
 *    , tools    => [#{name := binary(), description => binary(),
 *                     schema => map()}] | undefined
 *    }
 *
 * `tools` are inlined as a synthetic system addendum because
 * llama_chat_apply_template does not take a tools field. Models that
 * embed tool definitions in their template (llama-3.1, hermes-2,
 * qwen2.5) read them from the system block.
 */

/* Pull a binary value out of `Map[Key]`. Returns 1 with `bin` filled
 * on success, 0 if the key is missing or not a binary. The returned
 * `bin` points into a process-owned region; copy before unlocking
 * any cross-call resource. */
static int get_map_bin(ErlNifEnv *env, ERL_NIF_TERM map, const char *key,
                      ErlNifBinary *bin) {
    ERL_NIF_TERM v;
    ERL_NIF_TERM k = enif_make_atom(env, key);
    if (!enif_get_map_value(env, map, k, &v)) return 0;
    if (!enif_inspect_iolist_as_binary(env, v, bin)) return 0;
    return 1;
}

static void free_chat_msgs(struct llama_chat_message *msgs, int n) {
    for (int i = 0; i < n; i++) {
        if (msgs[i].role) enif_free((char *) msgs[i].role);
        if (msgs[i].content) enif_free((char *) msgs[i].content);
    }
}

/* Iterate over a list of message maps and fill `out_msgs` with
 * llama_chat_message structs. Each message is `#{role := ..., content := ...}`.
 * The role and content strings are allocated with enif_alloc and the
 * caller must free them via free_chat_msgs.
 *
 * On error the helper frees the role+content allocations it placed
 * into out[idx0..idx-1] before returning. The caller's free_chat_msgs
 * call (over its pre-call n_msgs range) does not overlap that range,
 * so no double-free is reachable.
 *
 * Returns the number of messages on success, -1 on bad input
 * (missing role/content key, role not iolist-binary), -2 on OOM,
 * or -3 when `content` is present but not iolist-binary
 * (Anthropic-style content blocks). The caller distinguishes -3
 * to surface {error, invalid_content} rather than badarg.
 */
static int build_chat_msgs_from_list(
    ErlNifEnv *env, ERL_NIF_TERM list,
    struct llama_chat_message *out, int max_out, int idx0
) {
    int idx = idx0;
    int err = 0;
    ERL_NIF_TERM head, tail = list;
    while (enif_get_list_cell(env, tail, &head, &tail)) {
        if (idx >= max_out) { err = -1; goto cleanup; }
        if (!enif_is_map(env, head)) { err = -1; goto cleanup; }
        ErlNifBinary role_bin, content_bin;
        if (!get_map_bin(env, head, "role", &role_bin)) { err = -1; goto cleanup; }
        ERL_NIF_TERM content_term;
        if (!enif_get_map_value(env, head, enif_make_atom(env, "content"),
                                &content_term)) {
            err = -1;
            goto cleanup;
        }
        if (!enif_inspect_iolist_as_binary(env, content_term, &content_bin)) {
            err = -3;
            goto cleanup;
        }
        char *role = enif_alloc(role_bin.size + 1);
        if (!role) { err = -2; goto cleanup; }
        memcpy(role, role_bin.data, role_bin.size);
        role[role_bin.size] = '\0';
        char *content = enif_alloc(content_bin.size + 1);
        if (!content) {
            enif_free(role);
            err = -2;
            goto cleanup;
        }
        memcpy(content, content_bin.data, content_bin.size);
        content[content_bin.size] = '\0';
        out[idx].role = role;
        out[idx].content = content;
        idx++;
    }
    return idx;

cleanup:
    free_chat_msgs(out + idx0, idx - idx0);
    return err;
}

/* Build a synthetic system content string that prepends the user-
 * supplied system text and renders tools as a textual list, so models
 * whose chat templates honour tool definitions in the system block
 * (llama-3.1+, hermes-2-pro, qwen2.5) see them. Caller frees with
 * enif_free.
 *
 * Returns the malloced string or NULL on OOM. `*out_len` is set to
 * the strlen for convenience. */
static char *build_system_content(ErlNifEnv *env, ERL_NIF_TERM request_map,
                                  size_t *out_len) {
    ErlNifBinary system_bin = {0};
    int has_system = get_map_bin(env, request_map, "system", &system_bin);

    ERL_NIF_TERM tools_term;
    int has_tools =
        enif_get_map_value(env, request_map, enif_make_atom(env, "tools"),
                           &tools_term)
        && enif_is_list(env, tools_term);

    if (!has_system && !has_tools) {
        if (out_len) *out_len = 0;
        return NULL;
    }

    /* Render: `<system>\n\nAvailable tools:\n  - name: description\n...` */
    size_t cap = 256;
    if (has_system) cap += system_bin.size;
    char *buf = enif_alloc(cap);
    if (!buf) return NULL;
    size_t pos = 0;
    if (has_system) {
        memcpy(buf + pos, system_bin.data, system_bin.size);
        pos += system_bin.size;
    }
    if (has_tools) {
        const char *header = (has_system ? "\n\nAvailable tools:\n" :
                                            "Available tools:\n");
        size_t header_len = strlen(header);
        if (pos + header_len + 1 > cap) {
            cap = (pos + header_len + 1) * 2;
            char *nbuf = enif_realloc(buf, cap);
            if (!nbuf) { enif_free(buf); return NULL; }
            buf = nbuf;
        }
        memcpy(buf + pos, header, header_len);
        pos += header_len;
        ERL_NIF_TERM head, tail = tools_term;
        while (enif_get_list_cell(env, tail, &head, &tail)) {
            if (!enif_is_map(env, head)) continue;
            ErlNifBinary name_bin, desc_bin;
            if (!get_map_bin(env, head, "name", &name_bin)) continue;
            int has_desc = get_map_bin(env, head, "description", &desc_bin);
            size_t needed = 4 + name_bin.size + 2 +
                             (has_desc ? desc_bin.size : 0) + 1;
            if (pos + needed + 1 > cap) {
                cap = (pos + needed + 1) * 2;
                char *nbuf = enif_realloc(buf, cap);
                if (!nbuf) { enif_free(buf); return NULL; }
                buf = nbuf;
            }
            memcpy(buf + pos, "  - ", 4); pos += 4;
            memcpy(buf + pos, name_bin.data, name_bin.size); pos += name_bin.size;
            if (has_desc) {
                memcpy(buf + pos, ": ", 2); pos += 2;
                memcpy(buf + pos, desc_bin.data, desc_bin.size); pos += desc_bin.size;
            }
            buf[pos++] = '\n';
        }
    }
    buf[pos] = '\0';
    if (out_len) *out_len = pos;
    return buf;
}

ERL_NIF_TERM nif_apply_chat_template(ErlNifEnv *env, int argc,
                                            const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_model_t *m;
    if (!enif_get_resource(env, argv[0], MODEL_RT, (void **) &m)) {
        return enif_make_badarg(env);
    }
    if (!enif_is_map(env, argv[1])) {
        return enif_make_badarg(env);
    }

    /* Read the messages list from the request. */
    ERL_NIF_TERM messages_term;
    if (!enif_get_map_value(env, argv[1],
                            enif_make_atom(env, "messages"), &messages_term)
        || !enif_is_list(env, messages_term)) {
        return enif_make_badarg(env);
    }
    unsigned msg_len;
    if (!enif_get_list_length(env, messages_term, &msg_len)) {
        return enif_make_badarg(env);
    }

    /* +1 for an optional synthetic system message at the front. */
    int max_msgs = (int) msg_len + 1;
    struct llama_chat_message *msgs =
        enif_alloc(sizeof(struct llama_chat_message) * (size_t) max_msgs);
    if (!msgs) return enif_make_tuple2(env, atom_error, atom_oom);
    memset(msgs, 0, sizeof(struct llama_chat_message) * (size_t) max_msgs);

    int n_msgs = 0;
    char *synthetic_system = build_system_content(env, argv[1], NULL);
    if (synthetic_system) {
        char *role = enif_alloc(7);
        if (!role) {
            enif_free(synthetic_system);
            enif_free(msgs);
            return enif_make_tuple2(env, atom_error, atom_oom);
        }
        memcpy(role, "system", 7);
        msgs[0].role = role;
        msgs[0].content = synthetic_system;
        n_msgs = 1;
    }

    int built = build_chat_msgs_from_list(
        env, messages_term, msgs, max_msgs, n_msgs);
    if (built < 0) {
        free_chat_msgs(msgs, n_msgs);
        enif_free(msgs);
        switch (built) {
            case -2: return enif_make_tuple2(env, atom_error, atom_oom);
            case -3: return enif_make_tuple2(env, atom_error, atom_invalid_content);
            default: return enif_make_badarg(env);
        }
    }
    n_msgs = built;

    pthread_mutex_lock(&m->mu);
    if (!m->model || m->release_pending) {
        pthread_mutex_unlock(&m->mu);
        free_chat_msgs(msgs, n_msgs);
        enif_free(msgs);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    const char *tmpl = erllama_safe_model_chat_template(m->model, NULL);
    if (!tmpl || tmpl[0] == '\0') {
        pthread_mutex_unlock(&m->mu);
        free_chat_msgs(msgs, n_msgs);
        enif_free(msgs);
        return enif_make_tuple2(env, atom_error, atom_no_template);
    }

    /* Render. Start with a 4 KiB buffer; grow on negative-needed-size. */
    int32_t buf_size = 4096;
    char *buf = enif_alloc((size_t) buf_size);
    if (!buf) {
        pthread_mutex_unlock(&m->mu);
        free_chat_msgs(msgs, n_msgs);
        enif_free(msgs);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    int32_t written = erllama_safe_chat_apply_template(
        tmpl, msgs, (size_t) n_msgs, true, buf, buf_size);
    /* Vendored `llama_chat_apply_template` returns the formatted-chat
     * size as a positive value even when `buf` was too small to hold
     * it — strncpy silently truncates and the function does not
     * indicate truncation via a negative return. Detect truncation by
     * comparing `written` against `buf_size` and retry with a grown
     * buffer. (A negative return means an unknown template, which we
     * bail on below; INT32_MIN is the C++-exception sentinel from
     * `erllama_safe_chat_apply_template`.) */
    if (written > 0 && written > buf_size) {
        if (written > (int32_t) ERLLAMA_MAX_TOKEN_TEXT) {
            pthread_mutex_unlock(&m->mu);
            free_chat_msgs(msgs, n_msgs);
            enif_free(msgs);
            enif_free(buf);
            return enif_make_tuple2(env, atom_error, atom_too_large);
        }
        enif_free(buf);
        buf_size = written + 16;
        buf = enif_alloc((size_t) buf_size);
        if (!buf) {
            pthread_mutex_unlock(&m->mu);
            free_chat_msgs(msgs, n_msgs);
            enif_free(msgs);
            return enif_make_tuple2(env, atom_error, atom_oom);
        }
        written = erllama_safe_chat_apply_template(
            tmpl, msgs, (size_t) n_msgs, true, buf, buf_size);
        /* Defensive: a stable template render must produce the same
         * size on the second call. If it suddenly claims more, bail
         * rather than feed a still-truncated buffer to tokenize. */
        if (written > buf_size) {
            pthread_mutex_unlock(&m->mu);
            free_chat_msgs(msgs, n_msgs);
            enif_free(msgs);
            enif_free(buf);
            return enif_make_tuple2(env, atom_error, atom_template_failed);
        }
    }
    if (written < 0) {
        pthread_mutex_unlock(&m->mu);
        free_chat_msgs(msgs, n_msgs);
        enif_free(msgs);
        enif_free(buf);
        return enif_make_tuple2(env, atom_error,
                                written == INT32_MIN ? atom_exception
                                                     : atom_template_failed);
    }

    /* Tokenise the rendered string. parse_special=true so chat-template
     * tokens (`<|user|>`, `<|im_start|>`, etc.) become their special
     * token ids rather than text fragments. */
    const struct llama_vocab *vocab = erllama_safe_model_get_vocab(m->model);
    if (!vocab) {
        pthread_mutex_unlock(&m->mu);
        free_chat_msgs(msgs, n_msgs);
        enif_free(msgs);
        enif_free(buf);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }

    /* See the matching comment in `nif_tokenize`: written + 8 is a
     * sound upper bound and clamping down forces unnecessary retries. */
    int32_t n_max = written + 8;
    if (n_max < 16) n_max = 16;
    llama_token *tokens = enif_alloc(sizeof(llama_token) * (size_t) n_max);
    if (!tokens) {
        pthread_mutex_unlock(&m->mu);
        free_chat_msgs(msgs, n_msgs);
        enif_free(msgs);
        enif_free(buf);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    int32_t n = erllama_safe_tokenize(vocab, buf, written, tokens, n_max,
                                      true, true);
    if (n < 0 && n != INT32_MIN) {
        int32_t needed = -n;
        if (needed > ERLLAMA_MAX_TOKENS) {
            pthread_mutex_unlock(&m->mu);
            free_chat_msgs(msgs, n_msgs);
            enif_free(msgs);
            enif_free(buf);
            enif_free(tokens);
            return enif_make_tuple2(env, atom_error, atom_too_large);
        }
        enif_free(tokens);
        tokens = enif_alloc(sizeof(llama_token) * (size_t) needed);
        if (!tokens) {
            pthread_mutex_unlock(&m->mu);
            free_chat_msgs(msgs, n_msgs);
            enif_free(msgs);
            enif_free(buf);
            return enif_make_tuple2(env, atom_error, atom_oom);
        }
        n = erllama_safe_tokenize(vocab, buf, written, tokens, needed,
                                  true, true);
    }
    pthread_mutex_unlock(&m->mu);
    free_chat_msgs(msgs, n_msgs);
    enif_free(msgs);
    enif_free(buf);
    if (n == INT32_MIN) {
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    if (n < 0) {
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_tokenize_failed);
    }
    /* See the matching post-success check in `nif_tokenize`. */
    if (n > ERLLAMA_MAX_TOKENS) {
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_too_large);
    }

    ERL_NIF_TERM list = enif_make_list(env, 0);
    for (int32_t i = n - 1; i >= 0; i--) {
        list = enif_make_list_cell(env, enif_make_int(env, tokens[i]), list);
    }
    enif_free(tokens);
    return enif_make_tuple2(env, atom_ok, list);
}
