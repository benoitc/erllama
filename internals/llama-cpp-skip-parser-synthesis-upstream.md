## Upstream prompt: `skip_parser_synthesis` opt-in for `common_chat_templates_apply_jinja`

Target: https://github.com/ggml-org/llama.cpp (base tag: `b10068`)

### What this is

Add a per-call flag on `common_chat_templates_inputs` that short-circuits
`common_chat_templates_apply_jinja` to return a `common_chat_params` with only
`.prompt` populated. Callers who cache the autoparser separately then only pay
for prompt rendering per request.

Motivation: chat servers driving multi-turn agent traffic call
`common_chat_templates_apply` once per turn. For the same
`(tools, tool_choice, parallel_tool_calls)` tuple, the synthesised PEG parser
is deterministic and reusable, but today it is rebuilt every call because
`autoparser::analyze_template` + `peg_generator::generate_parser` run inside
the same function that renders the prompt. On a hosted OpenAI-compatible
server this shows up as O(50-150 ms) of per-request wall time that a cache
can eliminate.

The flag is default-`false` and opt-in; behaviour is unchanged for every
existing caller.

### Change

Two files, ~10 lines total.

#### `common/chat.h`

Add one field on `common_chat_templates_inputs`:

```cpp
    bool                                  add_bos = false;
    bool                                  add_eos = false;
    bool                                  force_pure_content = false;
    // Skip the PEG parser synthesis step inside
    // common_chat_templates_apply_jinja and return a common_chat_params
    // with only `.prompt' set. Lets callers cache the synthesised
    // parser separately and reuse it across turns with the same
    // (tools, tool_choice, parallel_tool_calls) tuple.
    bool                                  skip_parser_synthesis = false;
};
```

#### `common/chat.cpp`

In `common_chat_templates_apply_jinja`, insert a render-only short-circuit
right after the `force_pure_content` block and before
`common_chat_try_specialized_template`:

```cpp
    if (inputs.force_pure_content) {
        // ... existing block ...
        return data;
    }

    if (inputs.skip_parser_synthesis) {
        common_chat_params data;
        data.prompt = common_chat_template_direct_apply_impl(tmpl, params);
        return data;
    }

    if (auto result = common_chat_try_specialized_template(tmpl, src, params)) {
        return *result;
    }
```

### Why this shape

- One-line struct addition; opt-in; zero risk to existing callers.
- Reuses the already-in-scope `common_chat_template_direct_apply_impl` path
  that `force_pure_content` uses for its own render.
- Sits above `common_chat_try_specialized_template` so the caller can
  render fast even for templates that would otherwise walk the
  specialized-template path.

### Testing

A caller that pairs the render-only call with a cached full-synthesis
`common_chat_params` (built once per tools tuple) round-trips identical
`.prompt` bytes and parses tool calls identically against
`chat_parse` — because the render side runs the same
`common_chat_template_direct_apply_impl` and the parse side is untouched.

### Prior art

`inputs.force_pure_content` already establishes the pattern of an opt-in
flag on the same struct that short-circuits before parser synthesis. This
is the same shape.

### Author note

The barrel_inference project has been running this patch against b9334
through b10068 without churn. Landing it upstream drops one carried
diff we'd otherwise rebase every bump.
