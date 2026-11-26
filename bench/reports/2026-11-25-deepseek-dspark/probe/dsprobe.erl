-module(dsprobe).
-export([
    plain/1,
    checkpoint/2,
    restore/3,
    many/2,
    load/1,
    capture/1,
    rows/2,
    head/2,
    feed/4,
    draft/5,
    top/2
]).
load(P) -> erlang:load_nif(P, 0).
capture(_) -> erlang:nif_error(nif_not_loaded).
rows(_, _) -> erlang:nif_error(nif_not_loaded).
head(_, _) -> erlang:nif_error(nif_not_loaded).
feed(_, _, _, _) -> erlang:nif_error(nif_not_loaded).
draft(_, _, _, _, _) -> erlang:nif_error(nif_not_loaded).
top(_, _) -> erlang:nif_error(nif_not_loaded).
plain(_) -> erlang:nif_error(nif_not_loaded).
checkpoint(_, _) -> erlang:nif_error(nif_not_loaded).
restore(_, _, _) -> erlang:nif_error(nif_not_loaded).
many(_, _) -> erlang:nif_error(nif_not_loaded).
