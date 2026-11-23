/* Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
 * See the LICENSE file at the project root. */

/* Rank-local tensor parallelism (experimental): one OS process per
 * rank, each with its own GPU backend, joined into the backend's
 * communicator.
 *
 * The communicator's bootstrap (`ggml_backend_comm_init_rank') calls a
 * callback that must gather `world_size * size' bytes in rank order
 * from every rank before the communicator exists. The callback runs
 * inside nif_tp_open on its dirty thread: it sends this rank's bytes
 * to an Erlang agent process (`{erllama_tp_bootstrap, Bridge, Round,
 * Bin}'), which exchanges them with the other ranks through a
 * coordinator and answers with nif_tp_bootstrap_reply/3. The callback
 * waits for that answer until the open's deadline, then fails, so no
 * rank waits forever and every rank either joins or fails.
 *
 * Inside a collective, the backend aborts the process on error
 * (NCCL_CHECK): ranks are separate OS processes so a failure ends one
 * rank, not the coordinator. */

#include "erllama_tp_nif.h"

#include <errno.h>
#include <pthread.h>
#include <string.h>
#include <time.h>

#include <cstdint>
#include <new>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <mutex>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "erllama_safe.h"

namespace {

ErlNifResourceType *TP_BRIDGE_RT = nullptr;
ErlNifResourceType *TP_RANK_RT = nullptr;

ERL_NIF_TERM mk_atom(ErlNifEnv *env, const char *name) {
    ERL_NIF_TERM a;
    if (enif_make_existing_atom(env, name, &a, ERL_NIF_UTF8)) {
        return a;
    }
    return enif_make_atom(env, name);
}

ERL_NIF_TERM mk_error(ErlNifEnv *env, const char *reason) {
    return enif_make_tuple2(env, mk_atom(env, "error"), mk_atom(env, reason));
}

ERL_NIF_TERM mk_ok(ErlNifEnv *env, ERL_NIF_TERM v) {
    return enif_make_tuple2(env, mk_atom(env, "ok"), v);
}

int64_t now_ms() {
    return (int64_t) enif_monotonic_time(ERL_NIF_MSEC);
}

/* ---------------------------------------------------------------- */
/* Bootstrap bridge                                                  */
/* ---------------------------------------------------------------- */

struct tp_bridge {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    ErlNifPid agent;
    int world = 0;
    /* Round being waited on (1, 2, ...); 0 before the first. */
    uint64_t round = 0;
    size_t expect = 0;
    bool have_reply = false;
    bool rejected = false;
    bool closed = false;
    std::vector<uint8_t> reply;
    int64_t deadline_ms = 0;
    /* The NIF call's env while the callback can run in it (it runs
     * synchronously inside that call). */
    ErlNifEnv *caller_env = nullptr;
    const char *error = "bootstrap_failed";
};

void bridge_dtor(ErlNifEnv *, void *obj) {
    auto *b = static_cast<tp_bridge *>(obj);
    pthread_cond_destroy(&b->cv);
    pthread_mutex_destroy(&b->mu);
    b->~tp_bridge();
}

tp_bridge *bridge_new(ErlNifPid agent, int world, int timeout_ms) {
    void *mem = enif_alloc_resource(TP_BRIDGE_RT, sizeof(tp_bridge));
    if (!mem) {
        return nullptr;
    }
    auto *b = new (mem) tp_bridge();
    pthread_mutex_init(&b->mu, nullptr);
    pthread_cond_init(&b->cv, nullptr);
    b->agent = agent;
    b->world = world;
    b->deadline_ms = now_ms() + timeout_ms;
    return b;
}

/* Wait on the condition variable until `deadline_ms' (monotonic) as a
 * realtime absolute timespec. Returns ETIMEDOUT past the deadline. */
int bridge_wait(tp_bridge *b) {
    int64_t left = b->deadline_ms - now_ms();
    if (left <= 0) {
        return ETIMEDOUT;
    }
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += left / 1000;
    ts.tv_nsec += (left % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) {
        ts.tv_sec += 1;
        ts.tv_nsec -= 1000000000L;
    }
    return pthread_cond_timedwait(&b->cv, &b->mu, &ts);
}

/* The communicator's bootstrap callback: this rank's `size' bytes out,
 * every rank's bytes (rank order) into `recv'. */
bool bootstrap_cb(void *user_data, const void *send_buf, void *recv_buf, size_t size) {
    auto *b = static_cast<tp_bridge *>(user_data);
    uint64_t round;
    pthread_mutex_lock(&b->mu);
    if (b->closed) {
        b->error = "closed";
        pthread_mutex_unlock(&b->mu);
        return false;
    }
    round = ++b->round;
    b->expect = size * (size_t) b->world;
    b->have_reply = false;
    b->rejected = false;
    pthread_mutex_unlock(&b->mu);

    ErlNifEnv *menv = enif_alloc_env();
    ERL_NIF_TERM bin;
    unsigned char *p = enif_make_new_binary(menv, size, &bin);
    if (size > 0) {
        memcpy(p, send_buf, size);
    }
    ERL_NIF_TERM msg = enif_make_tuple4(menv, mk_atom(menv, "erllama_tp_bootstrap"),
                                        enif_make_resource(menv, b), enif_make_uint64(menv, round), bin);
    int sent = enif_send(b->caller_env, &b->agent, menv, msg);
    enif_free_env(menv);
    if (!sent) {
        b->error = "agent_down";
        return false;
    }

    bool ok = false;
    pthread_mutex_lock(&b->mu);
    while (!b->have_reply && !b->rejected && !b->closed) {
        if (bridge_wait(b) == ETIMEDOUT) {
            b->error = "timeout";
            break;
        }
    }
    if (b->have_reply && !b->closed) {
        memcpy(recv_buf, b->reply.data(), b->expect);
        /* A failure after a good exchange is the backend's own (a
         * rank's checks, or the communicator's init). */
        b->error = "init_failed";
        ok = true;
    } else if (b->rejected) {
        b->error = "rejected";
    } else if (b->closed) {
        b->error = "closed";
    }
    pthread_mutex_unlock(&b->mu);
    return ok;
}

/* ---------------------------------------------------------------- */
/* Rank                                                              */
/* ---------------------------------------------------------------- */

struct tp_group;

struct tp_rank {
    pthread_mutex_t mu;
    ggml_backend_t backend = nullptr;
    void *comm = nullptr;
    ggml_backend_comm_free_t comm_free = nullptr;
    ggml_backend_comm_allreduce_tensor_t allreduce = nullptr;
    int rank = 0;
    int world = 0;
    /* The devices a model is loaded on (created on first use). */
    tp_group *group = nullptr;
    /* Models loaded on the group: the rank stays open while any lives. */
    std::atomic<int> models{0};
};

void group_free(tp_group *g);

void rank_release(tp_rank *r) {
    if (r->group) {
        group_free(r->group);
        r->group = nullptr;
    }
    if (r->comm && r->comm_free) {
        try {
            r->comm_free(r->comm);
        } catch (...) {
        }
    }
    r->comm = nullptr;
    if (r->backend) {
        try {
            ggml_backend_free(r->backend);
        } catch (...) {
        }
    }
    r->backend = nullptr;
}

void rank_dtor(ErlNifEnv *, void *obj) {
    auto *r = static_cast<tp_rank *>(obj);
    rank_release(r);
    pthread_mutex_destroy(&r->mu);
    r->~tp_rank();
}

/* The `idx'-th GPU device (integrated GPUs count), or null. */
ggml_backend_dev_t gpu_device(int idx) {
    int seen = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        enum ggml_backend_dev_type t = ggml_backend_dev_type(d);
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            if (seen == idx) {
                return d;
            }
            seen++;
        }
    }
    return nullptr;
}

void *reg_proc(ggml_backend_dev_t dev, const char *name) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    return reg ? ggml_backend_reg_get_proc_address(reg, name) : nullptr;
}

/* ---------------------------------------------------------------- */
/* The group's devices                                               */
/* ---------------------------------------------------------------- */
/*
 * A model is loaded with llama.cpp's own tensor split
 * (LLAMA_SPLIT_MODE_TENSOR) over `world' devices, one per rank, in
 * rank order. On this rank:
 *
 *   - the device at this rank's slot wraps the local GPU: everything is
 *     forwarded to it, except that its backends report this device, so
 *     the split backend asks this device's reg for its collectives;
 *   - the devices at the other slots are placeholders for the other
 *     ranks: their buffers reserve address space and no memory, their
 *     graphs compute nothing, and their tensors read as zeros.
 *
 * So this rank holds and computes only its own slices, in the split
 * backend's order, and every all-reduce the split backend asks for is
 * this rank's tensor through the rank's communicator (the other ranks
 * do the same at the same point).
 *
 * The communicator queues its work on the CUDA stream of the backend it
 * was created with, not the one the model computes on: events order the
 * two streams around each reduction.
 */

struct tp_dev_ctx {
    tp_group *g = nullptr;
    int slot = 0;
    std::string name;
    std::string desc;
};

struct tp_group {
    tp_rank *rank = nullptr;
    int world = 0;
    int slot = 0;
    ggml_backend_dev_t local = nullptr;
    ggml_backend_reg reg;
    std::vector<ggml_backend_device> devs;
    std::vector<tp_dev_ctx> dctx;
    ggml_backend_buffer_type ph_buft;
    /* Bytes the placeholders reserve (what the other ranks hold), and
     * the reductions made, with their sizes in order (bounded). */
    std::atomic<uint64_t> remote_bytes{0};
    std::atomic<uint64_t> reductions{0};
    std::mutex log_mu;
    std::vector<int64_t> sizes;
};

/* What the split backend's comm_init returns: the backend this rank
 * computes on and the events ordering it with the communicator. */
struct tp_comm {
    tp_group *g = nullptr;
    ggml_backend_t compute = nullptr;
    ggml_backend_event_t before = nullptr;
    ggml_backend_event_t after = nullptr;
};

constexpr size_t TP_LOG_MAX = 65536;

tp_dev_ctx *dc(ggml_backend_dev_t d) {
    return static_cast<tp_dev_ctx *>(d->context);
}

ggml_backend_dev_t local_of(ggml_backend_dev_t d) {
    return dc(d)->g->local;
}

/* --- the wrapper around the local GPU --- */

const char *w_name(ggml_backend_dev_t d) { return dc(d)->name.c_str(); }
const char *w_desc(ggml_backend_dev_t d) { return ggml_backend_dev_description(local_of(d)); }
void w_memory(ggml_backend_dev_t d, size_t *f, size_t *t) { ggml_backend_dev_memory(local_of(d), f, t); }
enum ggml_backend_dev_type w_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }
void w_props(ggml_backend_dev_t d, struct ggml_backend_dev_props *p) {
    ggml_backend_dev_get_props(local_of(d), p);
    p->name = dc(d)->name.c_str();
}
ggml_backend_t w_init(ggml_backend_dev_t d, const char *params) {
    ggml_backend_t b = ggml_backend_dev_init(local_of(d), params);
    if (b) {
        b->device = d;
    }
    return b;
}
ggml_backend_buffer_type_t w_buft(ggml_backend_dev_t d) { return ggml_backend_dev_buffer_type(local_of(d)); }
ggml_backend_buffer_type_t w_host_buft(ggml_backend_dev_t d) { return ggml_backend_dev_host_buffer_type(local_of(d)); }
ggml_backend_buffer_t w_from_ptr(ggml_backend_dev_t d, void *ptr, size_t size, size_t max) {
    return ggml_backend_dev_buffer_from_host_ptr(local_of(d), ptr, size, max);
}
bool w_supports_op(ggml_backend_dev_t d, const struct ggml_tensor *op) { return ggml_backend_dev_supports_op(local_of(d), op); }
bool w_supports_buft(ggml_backend_dev_t d, ggml_backend_buffer_type_t b) { return ggml_backend_dev_supports_buft(local_of(d), b); }
bool w_offload_op(ggml_backend_dev_t d, const struct ggml_tensor *op) { return ggml_backend_dev_offload_op(local_of(d), op); }
ggml_backend_event_t w_event_new(ggml_backend_dev_t d) { return ggml_backend_event_new(local_of(d)); }
void w_event_free(ggml_backend_dev_t, ggml_backend_event_t e) { ggml_backend_event_free(e); }
void w_event_sync(ggml_backend_dev_t, ggml_backend_event_t e) { ggml_backend_event_synchronize(e); }

const struct ggml_backend_device_i WRAPPER_I = {
    w_name, w_desc, w_memory, w_type, w_props, w_init, w_buft, w_host_buft, w_from_ptr,
    w_supports_op, w_supports_buft, w_offload_op, w_event_new, w_event_free, w_event_sync,
};

/* --- placeholders for the other ranks --- */

struct ph_buffer {
    void *base = nullptr;
    size_t map = 0;
    tp_group *g = nullptr;
};

void ph_free_buffer(ggml_backend_buffer_t b) {
    auto *pb = static_cast<ph_buffer *>(b->context);
    if (pb->base && pb->map) {
        munmap(pb->base, pb->map);
    }
    pb->g->remote_bytes -= b->size;
    delete pb;
}
void *ph_base(ggml_backend_buffer_t b) { return static_cast<ph_buffer *>(b->context)->base; }
void ph_memset(ggml_backend_buffer_t, struct ggml_tensor *, uint8_t, size_t, size_t) {}
void ph_set(ggml_backend_buffer_t, struct ggml_tensor *, const void *, size_t, size_t) {}
void ph_get(ggml_backend_buffer_t, const struct ggml_tensor *, void *data, size_t, size_t size) { memset(data, 0, size); }
bool ph_cpy(ggml_backend_buffer_t, const struct ggml_tensor *, struct ggml_tensor *) { return false; }
void ph_clear(ggml_backend_buffer_t, uint8_t) {}

const struct ggml_backend_buffer_i PH_BUFFER_I = {
    ph_free_buffer, ph_base, nullptr, ph_memset, ph_set, ph_get, nullptr, nullptr, ph_cpy, ph_clear,
    nullptr, nullptr, nullptr, nullptr,
};

tp_group *buft_group(ggml_backend_buffer_type_t buft) { return static_cast<tp_group *>(buft->context); }
ggml_backend_buffer_type_t local_buft(ggml_backend_buffer_type_t buft) {
    return ggml_backend_dev_buffer_type(buft_group(buft)->local);
}

const char *ph_buft_name(ggml_backend_buffer_type_t) { return "ERLLAMA_TP_REMOTE"; }
/* Address space only (PROT_NONE): nothing may read or write it, so a
 * stray access faults instead of reading garbage. */
ggml_backend_buffer_t ph_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    size_t page = (size_t) sysconf(_SC_PAGESIZE);
    size_t map = ((size > 0 ? size : 1) + page - 1) / page * page;
    void *base = mmap(nullptr, map, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (base == MAP_FAILED) {
        return nullptr;
    }
    auto *pb = new ph_buffer();
    pb->base = base;
    pb->map = map;
    pb->g = buft_group(buft);
    pb->g->remote_bytes += size;
    return ggml_backend_buffer_init(buft, PH_BUFFER_I, pb, size);
}
/* Same alignment, maximum and padding as the local GPU, so a
 * placeholder holds what the rank it stands for does. */
size_t ph_align(ggml_backend_buffer_type_t b) { return ggml_backend_buft_get_alignment(local_buft(b)); }
size_t ph_max(ggml_backend_buffer_type_t b) { return ggml_backend_buft_get_max_size(local_buft(b)); }
size_t ph_alloc_size(ggml_backend_buffer_type_t b, const struct ggml_tensor *t) {
    return ggml_backend_buft_get_alloc_size(local_buft(b), t);
}
bool ph_is_host(ggml_backend_buffer_type_t) { return false; }

const struct ggml_backend_buffer_type_i PH_BUFT_I = {
    ph_buft_name, ph_alloc, nullptr, ph_align, ph_max, ph_alloc_size, nullptr, ph_is_host,
};

const char *phb_name(ggml_backend_t b) { return dc(b->device)->name.c_str(); }
void phb_free(ggml_backend_t b) { delete b; }
void phb_set_async(ggml_backend_t, struct ggml_tensor *, const void *, size_t, size_t) {}
void phb_get_async(ggml_backend_t, const struct ggml_tensor *, void *data, size_t, size_t size) { memset(data, 0, size); }
void phb_sync(ggml_backend_t) {}
enum ggml_status phb_compute(ggml_backend_t, struct ggml_cgraph *) { return GGML_STATUS_SUCCESS; }

const struct ggml_backend_i PH_BACKEND_I = {
    phb_name, phb_free, phb_set_async, phb_get_async, nullptr, nullptr, nullptr, phb_sync,
    nullptr, nullptr, nullptr, nullptr, phb_compute, nullptr, nullptr, nullptr,
};

ggml_guid ph_guid = {0x65, 0x72, 0x6c, 0x6c, 0x61, 0x6d, 0x61, 0x5f, 0x74, 0x70, 0x5f, 0x72, 0x65, 0x6d, 0x6f, 0x74};

const char *p_desc(ggml_backend_dev_t d) { return dc(d)->desc.c_str(); }
void p_props(ggml_backend_dev_t d, struct ggml_backend_dev_props *p) {
    memset(p, 0, sizeof(*p));
    p->name = dc(d)->name.c_str();
    p->description = dc(d)->desc.c_str();
    p->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    ggml_backend_dev_memory(local_of(d), &p->memory_free, &p->memory_total);
}
ggml_backend_t p_init(ggml_backend_dev_t d, const char *) {
    return new ggml_backend{&ph_guid, PH_BACKEND_I, d, dc(d)};
}
ggml_backend_buffer_type_t p_buft(ggml_backend_dev_t d) { return &dc(d)->g->ph_buft; }
bool p_supports_op(ggml_backend_dev_t, const struct ggml_tensor *) { return true; }
bool p_supports_buft(ggml_backend_dev_t d, ggml_backend_buffer_type_t b) { return b == &dc(d)->g->ph_buft; }

const struct ggml_backend_device_i PLACEHOLDER_I = {
    w_name, p_desc, w_memory, w_type, p_props, p_init, p_buft, nullptr, nullptr,
    p_supports_op, p_supports_buft, nullptr, nullptr, nullptr, nullptr,
};

/* --- the reg: the split backend's collectives --- */

void *comm_init(ggml_backend_t *backends, size_t n) {
    if (n == 0 || !backends[0] || !backends[0]->device) {
        return nullptr;
    }
    tp_group *g = dc(backends[0]->device)->g;
    if ((int) n != g->world || !backends[g->slot]) {
        return nullptr;
    }
    auto *c = new tp_comm();
    c->g = g;
    c->compute = backends[g->slot];
    c->before = ggml_backend_event_new(g->local);
    c->after = ggml_backend_event_new(g->local);
    if (!c->before || !c->after) {
        if (c->before) ggml_backend_event_free(c->before);
        if (c->after) ggml_backend_event_free(c->after);
        delete c;
        return nullptr;
    }
    return c;
}

void comm_free(void *ctx) {
    auto *c = static_cast<tp_comm *>(ctx);
    if (!c) {
        return;
    }
    ggml_backend_event_free(c->before);
    ggml_backend_event_free(c->after);
    delete c;
}

bool comm_allreduce(void *ctx, struct ggml_tensor **tensors) {
    auto *c = static_cast<tp_comm *>(ctx);
    tp_group *g = c->g;
    tp_rank *r = g->rank;
    ggml_tensor *t = tensors[g->slot];
    ggml_backend_event_record(c->before, c->compute);
    ggml_backend_event_wait(r->backend, c->before);
    ggml_tensor *mine[1] = {t};
    bool ok = r->allreduce(r->comm, mine);
    ggml_backend_event_record(c->after, r->backend);
    ggml_backend_event_wait(c->compute, c->after);
    g->reductions++;
    {
        std::lock_guard<std::mutex> l(g->log_mu);
        if (g->sizes.size() < TP_LOG_MAX) {
            g->sizes.push_back(ggml_nelements(t));
        }
    }
    return ok;
}

const char *reg_name(ggml_backend_reg_t) { return "ERLLAMA_TP"; }
size_t reg_count(ggml_backend_reg_t r) { return static_cast<tp_group *>(r->context)->devs.size(); }
ggml_backend_dev_t reg_dev(ggml_backend_reg_t r, size_t i) { return &static_cast<tp_group *>(r->context)->devs[i]; }
void *tp_reg_proc(ggml_backend_reg_t, const char *name) {
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *) comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *) comm_allreduce;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *) comm_free;
    }
    return nullptr;
}

const struct ggml_backend_reg_i REG_I = {reg_name, reg_count, reg_dev, tp_reg_proc};

tp_group *group_new(tp_rank *r, ggml_backend_dev_t local) {
    auto *g = new tp_group();
    g->rank = r;
    g->world = r->world;
    g->slot = r->rank;
    g->local = local;
    g->reg = ggml_backend_reg{GGML_BACKEND_API_VERSION, REG_I, g};
    g->ph_buft = ggml_backend_buffer_type{PH_BUFT_I, nullptr, g};
    g->dctx.resize((size_t) g->world);
    g->devs.resize((size_t) g->world);
    for (int i = 0; i < g->world; i++) {
        tp_dev_ctx &d = g->dctx[(size_t) i];
        d.g = g;
        d.slot = i;
        d.name = "TP" + std::to_string(i);
        d.desc = i == g->slot ? std::string(ggml_backend_dev_description(local))
                              : "erllama tensor-parallel rank " + std::to_string(i);
        g->devs[(size_t) i] = ggml_backend_device{i == g->slot ? WRAPPER_I : PLACEHOLDER_I, &g->reg, &d};
    }
    /* Any placeholder: the one after this rank's slot. */
    g->ph_buft.device = &g->devs[(size_t) ((g->slot + 1) % g->world)];
    return g;
}

void group_free(tp_group *g) {
    delete g;
}

bool get_rank(ErlNifEnv *env, ERL_NIF_TERM t, tp_rank **out) {
    return enif_get_resource(env, t, TP_RANK_RT, (void **) out);
}

/* Upload `data' into a 1-D or 2-D F32 tensor on the rank's backend. */
struct tensor_scope {
    ggml_context *ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ~tensor_scope() {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

ggml_context *small_ctx(size_t n_tensors) {
    struct ggml_init_params p = {
        ggml_tensor_overhead() * n_tensors + ggml_graph_overhead(),
        nullptr,
        true,
    };
    return ggml_init(p);
}

/* All-reduce `t' in place (sum over ranks) and read it back. */
bool reduce_into(tp_rank *r, ggml_tensor *t, std::vector<float> &out) {
    t->flags |= GGML_TENSOR_FLAG_COMPUTE;
    ggml_tensor *tensors[1] = {t};
    if (!r->allreduce(r->comm, tensors)) {
        return false;
    }
    ggml_backend_synchronize(r->backend);
    out.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    return true;
}

ERL_NIF_TERM f32_binary(ErlNifEnv *env, const std::vector<float> &v) {
    ERL_NIF_TERM bin;
    unsigned char *p = enif_make_new_binary(env, v.size() * sizeof(float), &bin);
    if (!v.empty()) {
        memcpy(p, v.data(), v.size() * sizeof(float));
    }
    return bin;
}

} // namespace

/* ---------------------------------------------------------------- */
/* Entry points                                                      */
/* ---------------------------------------------------------------- */

extern "C" int tp_nif_load(ErlNifEnv *env) {
    TP_BRIDGE_RT = enif_open_resource_type(env, nullptr, "erllama_tp_bridge", bridge_dtor,
                                           ERL_NIF_RT_CREATE, nullptr);
    TP_RANK_RT = enif_open_resource_type(env, nullptr, "erllama_tp_rank", rank_dtor,
                                         ERL_NIF_RT_CREATE, nullptr);
    return (TP_BRIDGE_RT && TP_RANK_RT) ? 0 : -1;
}

/* nif_tp_caps() -> {ok, #{gpus, init_rank, allreduce}}: the number of
 * GPU devices, and whether the first one's backend can join a
 * per-process communicator (built with NCCL) and all-reduce. */
extern "C" ERL_NIF_TERM nif_tp_caps(ErlNifEnv *env, int, const ERL_NIF_TERM[]) {
    if (erllama_safe_backend_init_once() != 0) {
        return mk_error(env, "exception");
    }
    try {
        int gpus = 0;
        while (gpu_device(gpus)) {
            gpus++;
        }
        ggml_backend_dev_t dev = gpu_device(0);
        bool init_rank = dev && reg_proc(dev, "ggml_backend_comm_init_rank");
        bool allreduce = dev && reg_proc(dev, "ggml_backend_comm_allreduce_tensor");
        ERL_NIF_TERM m = enif_make_new_map(env);
        enif_make_map_put(env, m, mk_atom(env, "gpus"), enif_make_int(env, gpus), &m);
        enif_make_map_put(env, m, mk_atom(env, "init_rank"), mk_atom(env, init_rank ? "true" : "false"), &m);
        enif_make_map_put(env, m, mk_atom(env, "allreduce"), mk_atom(env, allreduce ? "true" : "false"), &m);
        if (dev) {
            enif_make_map_put(env, m, mk_atom(env, "device"),
                              enif_make_string(env, ggml_backend_dev_name(dev), ERL_NIF_UTF8), &m);
        }
        return mk_ok(env, m);
    } catch (...) {
        return mk_error(env, "exception");
    }
}

/* nif_tp_open(GpuIndex, Rank, World, AgentPid, TimeoutMs) ->
 *     {ok, RankRef} | {error, Reason}
 * Blocks (dirty IO) until every rank joined, the bootstrap failed or
 * the deadline passed. Reason: not_supported (no GPU, or a backend
 * without per-process communicators), timeout, agent_down, rejected
 * (the coordinator answered an error), init_failed. */
extern "C" ERL_NIF_TERM nif_tp_open(ErlNifEnv *env, int, const ERL_NIF_TERM argv[]) {
    int gpu, rank, world, timeout_ms;
    ErlNifPid agent;
    if (!enif_get_int(env, argv[0], &gpu) || !enif_get_int(env, argv[1], &rank) ||
        !enif_get_int(env, argv[2], &world) || !enif_get_local_pid(env, argv[3], &agent) ||
        !enif_get_int(env, argv[4], &timeout_ms) || world < 1 || rank < 0 || rank >= world ||
        timeout_ms < 0) {
        return enif_make_badarg(env);
    }
    if (erllama_safe_backend_init_once() != 0) {
        return mk_error(env, "exception");
    }
    try {
        ggml_backend_dev_t dev = gpu_device(gpu);
        if (!dev) {
            return mk_error(env, "not_supported");
        }
        auto init_rank = (ggml_backend_comm_init_rank_t) reg_proc(dev, "ggml_backend_comm_init_rank");
        auto allreduce =
            (ggml_backend_comm_allreduce_tensor_t) reg_proc(dev, "ggml_backend_comm_allreduce_tensor");
        auto comm_free = (ggml_backend_comm_free_t) reg_proc(dev, "ggml_backend_comm_free");
        if (!init_rank || !allreduce || !comm_free) {
            return mk_error(env, "not_supported");
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            return mk_error(env, "init_failed");
        }
        tp_bridge *b = bridge_new(agent, world, timeout_ms);
        if (!b) {
            ggml_backend_free(backend);
            return mk_error(env, "oom");
        }
        b->caller_env = env;
        void *comm = init_rank(backend, rank, world, bootstrap_cb, b);
        pthread_mutex_lock(&b->mu);
        b->caller_env = nullptr;
        b->closed = true;
        const char *err = b->round > 0 ? b->error : "init_failed";
        pthread_mutex_unlock(&b->mu);
        enif_release_resource(b);
        if (!comm) {
            ggml_backend_free(backend);
            return mk_error(env, err);
        }
        void *mem = enif_alloc_resource(TP_RANK_RT, sizeof(tp_rank));
        if (!mem) {
            comm_free(comm);
            ggml_backend_free(backend);
            return mk_error(env, "oom");
        }
        auto *r = new (mem) tp_rank();
        pthread_mutex_init(&r->mu, nullptr);
        r->backend = backend;
        r->comm = comm;
        r->comm_free = comm_free;
        r->allreduce = allreduce;
        r->rank = rank;
        r->world = world;
        ERL_NIF_TERM t = enif_make_resource(env, r);
        enif_release_resource(r);
        return mk_ok(env, t);
    } catch (...) {
        return mk_error(env, "exception");
    }
}

/* nif_tp_bootstrap_reply(Bridge, Round, AllBin | error) -> ok | {error, stale}
 * The agent's answer to `{erllama_tp_bootstrap, Bridge, Round, _}':
 * every rank's bytes in rank order, or `error' when the exchange
 * failed. A reply to a round no longer waited on is stale. */
extern "C" ERL_NIF_TERM nif_tp_bootstrap_reply(ErlNifEnv *env, int, const ERL_NIF_TERM argv[]) {
    tp_bridge *b;
    ErlNifUInt64 round;
    if (!enif_get_resource(env, argv[0], TP_BRIDGE_RT, (void **) &b) ||
        !enif_get_uint64(env, argv[1], &round)) {
        return enif_make_badarg(env);
    }
    ErlNifBinary bin;
    bool is_bin = enif_inspect_binary(env, argv[2], &bin);
    pthread_mutex_lock(&b->mu);
    if (b->closed || round != b->round || b->have_reply || b->rejected) {
        pthread_mutex_unlock(&b->mu);
        return mk_error(env, "stale");
    }
    if (is_bin && bin.size == b->expect) {
        b->reply.assign(bin.data, bin.data + bin.size);
        b->have_reply = true;
    } else {
        b->rejected = true;
    }
    pthread_cond_broadcast(&b->cv);
    pthread_mutex_unlock(&b->mu);
    return mk_atom(env, "ok");
}

/* nif_tp_matvec(Rank, Rows, ColsLocal, WBin, XBin) ->
 *     {ok, PartialBin, ReducedBin} | {error, Reason}
 * Computes y = W·x on this rank's GPU (W: Rows x ColsLocal F32, row
 * major; x: ColsLocal F32), reads the partial y, all-reduces it with
 * the other ranks, and reads the sum. */
extern "C" ERL_NIF_TERM nif_tp_matvec(ErlNifEnv *env, int, const ERL_NIF_TERM argv[]) {
    tp_rank *r;
    int rows, cols;
    ErlNifBinary w, x;
    if (!get_rank(env, argv[0], &r) || !enif_get_int(env, argv[1], &rows) ||
        !enif_get_int(env, argv[2], &cols) || !enif_inspect_binary(env, argv[3], &w) ||
        !enif_inspect_binary(env, argv[4], &x) || rows < 1 || cols < 1 ||
        w.size != (size_t) rows * cols * sizeof(float) || x.size != (size_t) cols * sizeof(float)) {
        return enif_make_badarg(env);
    }
    pthread_mutex_lock(&r->mu);
    if (!r->comm) {
        pthread_mutex_unlock(&r->mu);
        return mk_error(env, "closed");
    }
    ERL_NIF_TERM result;
    try {
        tensor_scope s;
        s.ctx = small_ctx(4);
        ggml_tensor *tw = ggml_new_tensor_2d(s.ctx, GGML_TYPE_F32, cols, rows);
        ggml_tensor *tx = ggml_new_tensor_2d(s.ctx, GGML_TYPE_F32, cols, 1);
        ggml_tensor *ty = ggml_mul_mat(s.ctx, tw, tx);
        ggml_cgraph *gf = ggml_new_graph(s.ctx);
        ggml_build_forward_expand(gf, ty);
        s.buf = ggml_backend_alloc_ctx_tensors(s.ctx, r->backend);
        if (!s.buf) {
            pthread_mutex_unlock(&r->mu);
            return mk_error(env, "oom");
        }
        ggml_backend_tensor_set(tw, w.data, 0, w.size);
        ggml_backend_tensor_set(tx, x.data, 0, x.size);
        if (ggml_backend_graph_compute(r->backend, gf) != GGML_STATUS_SUCCESS) {
            pthread_mutex_unlock(&r->mu);
            return mk_error(env, "compute_failed");
        }
        std::vector<float> partial((size_t) rows);
        ggml_backend_tensor_get(ty, partial.data(), 0, ggml_nbytes(ty));
        std::vector<float> reduced;
        if (!reduce_into(r, ty, reduced)) {
            result = mk_error(env, "reduce_failed");
        } else {
            result = enif_make_tuple3(env, mk_atom(env, "ok"), f32_binary(env, partial),
                                      f32_binary(env, reduced));
        }
    } catch (...) {
        result = mk_error(env, "exception");
    }
    pthread_mutex_unlock(&r->mu);
    return result;
}

/* nif_tp_reduce(Rank, F32Bin) -> {ok, SumBin} | {error, Reason}
 * All-reduces the given values with the other ranks (the backend picks
 * FP32 or BF16 by size, as for a graph's partial sums). */
extern "C" ERL_NIF_TERM nif_tp_reduce(ErlNifEnv *env, int, const ERL_NIF_TERM argv[]) {
    tp_rank *r;
    ErlNifBinary in;
    if (!get_rank(env, argv[0], &r) || !enif_inspect_binary(env, argv[1], &in) || in.size == 0 ||
        in.size % sizeof(float) != 0) {
        return enif_make_badarg(env);
    }
    pthread_mutex_lock(&r->mu);
    if (!r->comm) {
        pthread_mutex_unlock(&r->mu);
        return mk_error(env, "closed");
    }
    ERL_NIF_TERM result;
    try {
        tensor_scope s;
        s.ctx = small_ctx(1);
        ggml_tensor *t = ggml_new_tensor_1d(s.ctx, GGML_TYPE_F32, (int64_t) (in.size / sizeof(float)));
        s.buf = ggml_backend_alloc_ctx_tensors(s.ctx, r->backend);
        if (!s.buf) {
            pthread_mutex_unlock(&r->mu);
            return mk_error(env, "oom");
        }
        ggml_backend_tensor_set(t, in.data, 0, in.size);
        std::vector<float> out;
        result = reduce_into(r, t, out) ? mk_ok(env, f32_binary(env, out)) : mk_error(env, "reduce_failed");
    } catch (...) {
        result = mk_error(env, "exception");
    }
    pthread_mutex_unlock(&r->mu);
    return result;
}

/* nif_tp_close(Rank) -> ok: frees the communicator and the backend
 * (idempotent; the resource destructor does it otherwise). */
extern "C" ERL_NIF_TERM nif_tp_close(ErlNifEnv *env, int, const ERL_NIF_TERM argv[]) {
    tp_rank *r;
    if (!get_rank(env, argv[0], &r)) {
        return enif_make_badarg(env);
    }
    pthread_mutex_lock(&r->mu);
    if (r->models > 0) {
        pthread_mutex_unlock(&r->mu);
        return mk_error(env, "in_use");
    }
    rank_release(r);
    pthread_mutex_unlock(&r->mu);
    return mk_atom(env, "ok");
}

/* nif_tp_bootstrap_selftest(World, PayloadBin, Rounds, AgentPid, TimeoutMs)
 *     -> {ok, [AllBin]} | {error, Reason}
 * Runs the bootstrap callback `Rounds' times with `PayloadBin' as this
 * rank's bytes, exactly as a communicator's init would, without a GPU:
 * the portable test of the bridge, the agent and the coordinator. */
extern "C" ERL_NIF_TERM nif_tp_bootstrap_selftest(ErlNifEnv *env, int, const ERL_NIF_TERM argv[]) {
    int world, rounds, timeout_ms;
    ErlNifBinary payload;
    ErlNifPid agent;
    if (!enif_get_int(env, argv[0], &world) || !enif_inspect_binary(env, argv[1], &payload) ||
        !enif_get_int(env, argv[2], &rounds) || !enif_get_local_pid(env, argv[3], &agent) ||
        !enif_get_int(env, argv[4], &timeout_ms) || world < 1 || rounds < 1 || timeout_ms < 0) {
        return enif_make_badarg(env);
    }
    std::vector<uint8_t> send(payload.data, payload.data + payload.size);
    tp_bridge *b = bridge_new(agent, world, timeout_ms);
    if (!b) {
        return mk_error(env, "oom");
    }
    b->caller_env = env;
    ERL_NIF_TERM list = enif_make_list(env, 0);
    std::vector<ERL_NIF_TERM> all;
    const char *err = nullptr;
    std::vector<uint8_t> recv(send.size() * (size_t) world);
    for (int i = 0; i < rounds; i++) {
        if (!bootstrap_cb(b, send.data(), recv.data(), send.size())) {
            err = b->error;
            break;
        }
        ERL_NIF_TERM bin;
        unsigned char *p = enif_make_new_binary(env, recv.size(), &bin);
        if (!recv.empty()) {
            memcpy(p, recv.data(), recv.size());
        }
        all.push_back(bin);
    }
    pthread_mutex_lock(&b->mu);
    b->caller_env = nullptr;
    b->closed = true;
    pthread_mutex_unlock(&b->mu);
    enif_release_resource(b);
    if (err) {
        return mk_error(env, err);
    }
    for (size_t i = all.size(); i > 0; i--) {
        list = enif_make_list_cell(env, all[i - 1], list);
    }
    return mk_ok(env, list);
}

/* ---------------------------------------------------------------- */
/* For the model loader and the decode NIF (C callers)               */
/* ---------------------------------------------------------------- */

/* The devices to load a model on this rank's group: `world' entries,
 * in rank order, then NULL (`cap' must leave room). Keeps the rank (and
 * so its communicator and devices) alive until
 * erllama_tp_model_released(*keep). Returns the device count, or -1. */
extern "C" int erllama_tp_model_devices(ErlNifEnv *env, ERL_NIF_TERM term, ggml_backend_dev_t *out,
                                        size_t cap, void **keep) {
    tp_rank *r;
    if (!get_rank(env, term, &r)) {
        return -1;
    }
    pthread_mutex_lock(&r->mu);
    if (!r->comm || (size_t) r->world + 1 > cap) {
        pthread_mutex_unlock(&r->mu);
        return -1;
    }
    try {
        if (!r->group) {
            r->group = group_new(r, ggml_backend_get_device(r->backend));
        }
    } catch (...) {
        pthread_mutex_unlock(&r->mu);
        return -1;
    }
    for (int i = 0; i < r->world; i++) {
        out[i] = &r->group->devs[(size_t) i];
    }
    out[r->world] = nullptr;
    r->models++;
    enif_keep_resource(r);
    *keep = r;
    pthread_mutex_unlock(&r->mu);
    return r->world;
}

extern "C" void erllama_tp_model_released(void *keep) {
    auto *r = static_cast<tp_rank *>(keep);
    r->models--;
    enif_release_resource(r);
}

/* Sum `n' floats on the host over every rank, exactly: in chunks below
 * the backend's FP32 threshold (larger buffers are reduced in BF16).
 * Every rank must call it with the same `n'. 0 on success. */
extern "C" int erllama_tp_reduce_host(void *keep, float *data, size_t n) {
    constexpr size_t CHUNK = 32000;
    auto *r = static_cast<tp_rank *>(keep);
    pthread_mutex_lock(&r->mu);
    int rc = 0;
    try {
        tensor_scope s;
        /* The chunk tensor, and one view per chunk. */
        s.ctx = small_ctx(2 + n / CHUNK);
        ggml_tensor *t = ggml_new_tensor_1d(s.ctx, GGML_TYPE_F32, (int64_t) CHUNK);
        s.buf = ggml_backend_alloc_ctx_tensors(s.ctx, r->backend);
        if (!s.buf || !r->comm) {
            rc = -1;
        }
        for (size_t off = 0; rc == 0 && off < n; off += CHUNK) {
            size_t len = std::min(CHUNK, n - off);
            ggml_tensor *v = ggml_view_1d(s.ctx, t, (int64_t) len, 0);
            v->buffer = t->buffer;
            v->data = t->data;
            ggml_backend_tensor_set(v, data + off, 0, len * sizeof(float));
            std::vector<float> out;
            if (!reduce_into(r, v, out)) {
                rc = -1;
                break;
            }
            memcpy(data + off, out.data(), len * sizeof(float));
        }
    } catch (...) {
        rc = -1;
    }
    pthread_mutex_unlock(&r->mu);
    return rc;
}

/* nif_tp_stats(Rank) -> {ok, #{remote_bytes, reductions, sizes}}: what
 * the placeholders reserve (the other ranks' share of every buffer),
 * and the model's reductions so far with their element counts, in
 * order (the first 65536). */
extern "C" ERL_NIF_TERM nif_tp_stats(ErlNifEnv *env, int, const ERL_NIF_TERM argv[]) {
    tp_rank *r;
    if (!get_rank(env, argv[0], &r)) {
        return enif_make_badarg(env);
    }
    pthread_mutex_lock(&r->mu);
    ERL_NIF_TERM m = enif_make_new_map(env);
    uint64_t remote = 0, reductions = 0;
    std::vector<int64_t> sizes;
    if (r->group) {
        remote = r->group->remote_bytes;
        reductions = r->group->reductions;
        std::lock_guard<std::mutex> l(r->group->log_mu);
        sizes = r->group->sizes;
    }
    pthread_mutex_unlock(&r->mu);
    ERL_NIF_TERM list = enif_make_list(env, 0);
    for (size_t i = sizes.size(); i > 0; i--) {
        list = enif_make_list_cell(env, enif_make_int64(env, sizes[i - 1]), list);
    }
    enif_make_map_put(env, m, mk_atom(env, "remote_bytes"), enif_make_uint64(env, remote), &m);
    enif_make_map_put(env, m, mk_atom(env, "reductions"), enif_make_uint64(env, reductions), &m);
    enif_make_map_put(env, m, mk_atom(env, "sizes"), list, &m);
    return mk_ok(env, m);
}
