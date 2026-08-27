# Vision and audio input

Sending images and audio to multimodal models through the mmproj
projector (libmtmd). You need this page when your model ships a
companion `mmproj-*.gguf` and you want to ask questions about
pictures or recordings: vision models (SmolVLM, Gemma 3 vision,
Qwen2-VL, InternVL, ...) and audio-input models (Qwen2-Audio,
Voxtral, Ultravox).

This is media INPUT to an LLM. It is not a transcription engine: a
dedicated Whisper-style ASR API (plain Whisper GGUFs, timestamps,
segments) is a different runtime (whisper.cpp) and stays on the
roadmap as its own item. An audio-capable model here can transcribe
when you ask it to, but as chat output.

## Load the projector

A multimodal model is two GGUF files: the text model plus the
projector that encodes images/audio into embeddings. Pass both:

```erlang
{ok, M} = erllama:load_model(<<"smolvlm">>, #{
    model_path => "/srv/models/SmolVLM-256M-Instruct-Q8_0.gguf",
    mmproj_path => "/srv/models/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"
}).
```

`mmproj_opts => #{use_gpu, n_threads, image_min_tokens,
image_max_tokens}` tunes the projector. Feature-detect through
`model_info/1`:

```erlang
{ok, #{mmproj := #{vision := true, audio := false}}} = erllama:model_info(M).
```

## Chat with images or audio

Message content may be a list of parts in the OpenAI shape; the
chat template places the media at the right position:

```erlang
{ok, Jpg} = file:read_file("photo.jpg"),
{ok, #{message := Msg}} = erllama:chat(M, [
    #{role => user, content => [
        #{type => text, text => <<"What is in this picture?">>},
        #{type => image, data => Jpg}
    ]}
], #{response_tokens => 128}).
```

Audio parts work the same way on audio-capable projectors:
`#{type => audio, data => WavBytes}`. Formats: images as jpg, png,
bmp or gif; audio as wav, mp3 or flac. Decoding happens inside the
engine; you pass the encoded file bytes.

## Raw prompts

`stream/3` and `complete/3` take a `media` option when you render
the prompt yourself. Put one marker per item where its tokens
belong (`erllama:media_marker/0` returns it):

```erlang
Prompt = <<"<|im_start|>User: <__media__>Describe this.<end_of_utterance>\nAssistant:">>,
{ok, Ref} = erllama:stream(M, Prompt, #{
    response_tokens => 64,
    media => [#{type => image, data => Jpg}]
}),
{ok, #{reply := Reply, stats := Stats}} = erllama:collect(Ref, 60000),
#{items := 1, n_tokens := _, n_pos := _} = maps:get(media, Stats).
```

A marker-count mismatch fails the request with
`{error, {media_failed, _}}`; the model recovers for the next
request. Stats report `media => #{items, n_tokens, n_pos}`
(`n_pos` can differ from `n_tokens` on M-RoPE models such as
Qwen2-VL).

## What media requests cannot do (v1)

- **No KV-cache reuse.** Media requests bypass cache lookup and
  saves entirely: two different images render byte-identical
  prompts, so reuse would need media-aware keys (a follow-up).
  Every media request prefills cold.
- **No sessions.** `session_id`, `parent_key`, `expect_committed`,
  `speculative` and `prefix_checkpoint_len` are rejected with
  `{error, {unsupported_with_media, Key}}`; `continue/3` rejects
  media outright.
- **Sole-request prefill.** The media evaluation cannot co-batch:
  while other requests are active on the model, a media request
  waits its turn (generation afterwards co-batches normally).
- **No video, webp, or TTS.** Video/webp need an ffmpeg subprocess
  (disabled in the NIF build); mtmd's experimental audio generation
  is not exposed.

## Notes

- The prompt for a media request must be a binary (token lists
  cannot carry marker positions).
- A model loaded without `mmproj_path` rejects media with
  `{error, no_mmproj}`; an image-only projector rejects audio parts
  with `{error, {unsupported_media, audio}}`.
- The per-step decode deadline does not bound the media prefill
  (it spans several internal decodes); `erllama:cancel/1` still
  interrupts it.

## See also

- [Loading a model](loading.md) - `mmproj_path`, `mmproj_opts`
- [Tool calls](tool-calls.md) - chat message shapes
- [Generating text](generation.md) - streaming, options, stats
- [Testing](testing.md) - the vision suite's environment variables
