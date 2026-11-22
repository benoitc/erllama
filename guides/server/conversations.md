# Conversations

A conversation is the history the release keeps for `erllama run`,
`erllama chat`, an agent session, or a stored OpenAI Responses chain.
Conversations are kept on disk and survive a restart of the release.
You need this page to find, continue, copy, export or delete one, and
to understand what happens to a tool call when the release stops in
the middle of it.

## List and read

```bash
R=_build/prod/rel/erllama_server
$R/bin/erllama conversations
conv_7f3a...  qwen  12 messages  2026-11-22T10:04:51Z
conv_19be...  qwen  4 messages  2026-11-22T09:58:02Z  running agt_51c0...
$R/bin/erllama conversation show conv_7f3a...
```

`show` prints the conversation as Markdown, one section per message.
A conversation with unresolved tool calls says so (see below).

## Continue

```bash
$R/bin/erllama chat qwen --conversation conv_7f3a...
$R/bin/erllama agent qwen --conversation conv_19be...
```

- This works after a restart too. The next turn starts from where
  the cache stood after the last one when the cache still holds it,
  and prefills the history otherwise.
- An agent session reuses the workspace its conversation was last
  worked in when you leave out `--workspace`.
- `erllama conversation attach ID` follows the run or session that
  is working in the conversation right now.

## Copy, export, delete

| Command | Does |
|---|---|
| `erllama conversation fork ID` | a new conversation with the same history; the two then go their own ways |
| `erllama conversation export ID` | the conversation as JSON (`--format md` for Markdown) |
| `erllama conversation rm ID` | deletes it; refused while a run or a session is working in it |

Through the service API: `conversations/0`, `conversation/1`,
`conversation_fork/1`, `conversation_export/2`,
`conversation_delete/1`.

## Where they are kept

- Each conversation is one file, `<data_dir>/conversations/<id>.jsonl`.
  Every change is written there before it takes effect.
- Stored responses (`/v1/responses` with `store: true`) are in
  `<data_dir>/responses/`. `previous_response_id` works after a
  restart.
- These files are separate from the KV cache. Clearing the cache never
  loses a conversation; it only makes the next turn start cold.

## A tool call cut off by a stop

An agent session records each tool call before it runs and its result
once it has finished, before the model goes on. When the release stops
or crashes, or a session dies, in the middle of a call:

- a call that had finished keeps its result;
- a call that was running is marked **unresolved**: the model is told
  its outcome is unknown, and the call is never run again, since it
  may have changed something already;
- a call that had not started is reported as not run.

`erllama conversations` and `conversation show` list unresolved calls,
so you can check the workspace yourself before you continue.
