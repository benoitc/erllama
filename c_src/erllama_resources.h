/* Shared private header exposing the NIF resource types defined in
 * erllama_nif.c so additional translation units (notably
 * erllama_chat_nif.cpp wrapping the llama.cpp autoparser) can
 * `enif_get_resource' against them and reach the underlying llama.cpp
 * pointers. The C file owns the definitions; this header is the
 * declaration surface.
 *
 * The resource pointers were `static' until this header landed. They
 * are now extern symbols set during NIF load. The dependent layouts
 * use only C-compatible types so this header is C/C++ safe behind
 * the extern "C" guard. */
#ifndef ERLLAMA_RESOURCES_H
#define ERLLAMA_RESOURCES_H

#include <pthread.h>
#include "erl_nif.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations from llama.cpp; struct definitions live in
 * <llama.h>. Including <llama.h> here would force every TU that
 * pulls this header to compile against the llama public API; the
 * forward decl is enough for the struct pointer fields below. */
struct llama_model;
struct llama_context;
struct llama_adapter_lora;
struct llama_sampler;

#define ERLLAMA_MAX_DEVICES 16

/* The canonical erllama_model_t definition (the NIF C files get it
 * through this header). Held by MODEL_RT resources. */
typedef struct {
    pthread_mutex_t mu;
    int mu_inited;
    struct llama_model *model;
    int active_contexts;
    int active_adapters;
    int release_pending;
    float tensor_split[ERLLAMA_MAX_DEVICES];
    int has_tensor_split;
} erllama_model_t;

extern ErlNifResourceType *MODEL_RT;
extern ErlNifResourceType *CTX_RT;
extern ErlNifResourceType *ADAPTER_RT;
extern ErlNifResourceType *SAMPLER_RT;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ERLLAMA_RESOURCES_H */
