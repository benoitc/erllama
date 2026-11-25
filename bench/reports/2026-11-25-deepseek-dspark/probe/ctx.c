#include "erllama_nif_int.h"
#include "erllama_resources.h"

void *dspark_lock(ErlNifEnv *env, ERL_NIF_TERM term,
                  struct llama_context **ctx) {
  erllama_context_t *r;
  if (!enif_get_resource(env, term, CTX_RT, (void **)&r))
    return NULL;
  pthread_mutex_lock(&r->mu);
  if (!r->ctx) {
    pthread_mutex_unlock(&r->mu);
    return NULL;
  }
  *ctx = r->ctx;
  return r;
}
void dspark_unlock(void *r) {
  pthread_mutex_unlock(&((erllama_context_t *)r)->mu);
}
void *dspark_rank(void *r) {
  erllama_context_t *c = r;
  return c->model_res ? c->model_res->tp_rank : NULL;
}
