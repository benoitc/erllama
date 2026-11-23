# A walkthrough: one release, many clients

This page runs the release through the things it is built for: one
loaded model shared by several conversations, an agent session working
on a slow command, a client that leaves and another that comes back,
the runtime view, one run cancelled without touching the others, and a
restart after which a conversation goes on. Run it to see the release's
behaviour for yourself, or to check it after a change.

There are two ways to run it:

- `scripts/demo.sh MODEL.gguf` drives a release on a real model with
  the `erllama` CLI. The model's answers vary from run to run; the
  release's behaviour does not.
- `erllama_server_demo_SUITE` runs the same story on the stub model
  with a scripted agent, and checks every step: `rebar3 ct --suite
  erllama_server_demo_SUITE --dir apps/erllama_server/test`.

## Run it

```bash
rebar3 as prod release
scripts/demo.sh ~/Models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

The script starts its own release with a scratch data directory and
stops it at the end; stop any release you already run first.

## What you see

On an Apple M4 Pro with Qwen2.5 1.5B Instruct Q4_K_M (output trimmed):

```text
== 1. one release, one model, four sequences
loaded demo:latest

== 2. two conversations, each with its own run
run A run_1004b9892a734edc, run B run_ce4573be6ce0849a

== 3. an agent session asked to run a slow command

== 4. meanwhile, another conversation
Paris
[took 0 s]

== 5. a client follows run A, leaves, and another comes back to it
the first client saw 123 lines, then went away
64
65
66
[the second client left too; run A goes on]

== 6. what runs, what waits, what the agent does
models
  demo:latest  running  ctx 8192  2 of 4 sequences busy
runs      2 running: 2 generating
agents
  agent_d65f1d43c1caaef7  tools  conv_5f134fc1dd11eca2, 1 tool call(s) running
...
turn          1 of 20 rounds, 26 of 32768 tokens, 594s left
tools
  shell call_b249dd6447557d24 (running)

== 7. cancel run B; run A goes on to its end
run_1004b9892a734edc  run  generating  demo:latest  conv_544510b367d80c1a
run_ce4573be6ce0849a  run  cancelled  demo:latest  conv_a4995cae04686b55
[cancelled, 47 prompt tokens, 523 generated]
[length, 45 prompt tokens, 2500 generated]
[tool] shell {"command":"sleep 20; echo slow done"}
[ok] exit 0
The command `sleep 20; echo slow done` was executed, and it printed "slow done" ...

== 8. restart the release and go on with conversation A
conv_544510b367d80c1a  demo:latest  2 messages  2026-11-23T07:47:39Z
...
[loading demo:latest]
647
```

What each step shows:

1. One release loads one model with four sequences; every client below
   uses it.
2. Two runs start in two conversations and return at once
   (`--detach`); they generate on the release, not in the terminal.
3. An agent session runs a 20-second command; the CLI that asked for it
   is a client like any other.
4. Another conversation gets its answer while the two runs generate and
   the agent waits on its command.
5. A client following run A is killed, as a closed terminal; another
   attaches from event 200 and reads on from there. Run A never
   noticed.
6. `erllama status` and `erllama inspect` show the two runs, the
   sequences they hold, and the session waiting on its shell call, with
   the turn's budget use.
7. Run B is cancelled; run A goes on to its token limit and the agent
   session finishes its turn.
8. The release stops and starts again. The conversations are still
   there; conversation A takes a new turn on top of its history (the
   model answers 647, the last number run A wrote before its limit).

## Limits

- The runs share the model: while four requests decode at once, each
  is slower than alone. Step 4's answer comes quickly because the
  question is short, not because it has the model to itself.
- Step 5's second client can only go back as far as the release's event
  buffer (`run_buffer_events`, 1024 events); asking for older events
  reports a gap and continues from the oldest one kept.
- Step 8 resumes the conversation, not the computation: a run or a
  tool call cut by a restart is not continued, and a tool call that
  may have run is marked unresolved and never run again.
- A crash in native code (llama.cpp, the NIF) would end the whole VM,
  every step at once. See [Operating the release](operations.md).
