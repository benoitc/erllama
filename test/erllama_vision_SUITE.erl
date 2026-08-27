%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Multimodal end-to-end against a real vision model + mmproj
%% projector (libmtmd). Gated on TWO environment variables, following
%% erllama_family_SUITE's pattern:
%%
%%   LLAMA_TEST_VISION_MODEL  - the text GGUF (e.g. SmolVLM-256M)
%%   LLAMA_TEST_MMPROJ        - its mmproj projector GGUF
%%
%% Optional third pair for audio-capable projectors:
%%
%%   LLAMA_TEST_AUDIO_MODEL / LLAMA_TEST_AUDIO_MMPROJ
%%
%% The test image is a synthesized BMP gradient (no fixture files).
-module(erllama_vision_SUITE).

-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([
    all/0,
    init_per_suite/1,
    end_per_suite/1,
    init_per_testcase/2,
    end_per_testcase/2
]).

-export([
    mmproj_caps_in_model_info/1,
    chat_with_image_generates/1,
    raw_stream_with_marker/1,
    marker_count_mismatch_fails/1,
    media_session_rejected/1,
    sequential_media_requests/1,
    audio_chat_generates/1
]).

-define(VISION_ENV, "LLAMA_TEST_VISION_MODEL").
-define(MMPROJ_ENV, "LLAMA_TEST_MMPROJ").
-define(AUDIO_ENV, "LLAMA_TEST_AUDIO_MODEL").
-define(AUDIO_MMPROJ_ENV, "LLAMA_TEST_AUDIO_MMPROJ").

all() ->
    [
        mmproj_caps_in_model_info,
        chat_with_image_generates,
        raw_stream_with_marker,
        marker_count_mismatch_fails,
        media_session_rejected,
        sequential_media_requests,
        audio_chat_generates
    ].

init_per_suite(Config) ->
    case {os:getenv(?VISION_ENV), os:getenv(?MMPROJ_ENV)} of
        {Model, Mmproj} when is_list(Model), is_list(Mmproj), Model =/= "", Mmproj =/= "" ->
            case filelib:is_regular(Model) andalso filelib:is_regular(Mmproj) of
                true ->
                    {ok, _} = application:ensure_all_started(erllama),
                    [{vision_model, Model}, {mmproj, Mmproj} | Config];
                false ->
                    {skip, "vision model or mmproj path is not a file"}
            end;
        _ ->
            {skip, "set " ?VISION_ENV " and " ?MMPROJ_ENV " to GGUF paths to enable this suite"}
    end.

end_per_suite(_Config) ->
    application:stop(erllama),
    ok.

init_per_testcase(TC, Config) ->
    PrivDir = ?config(priv_dir, Config),
    Dir = filename:join(PrivDir, atom_to_list(TC) ++ "_dir"),
    ok = filelib:ensure_path(Dir),
    DiskSrv = list_to_atom("vision_disk_" ++ atom_to_list(TC)),
    {ok, _} = erllama_cache_disk_srv:start_link(DiskSrv, Dir),
    Model = iolist_to_binary(["vision_model_", atom_to_binary(TC)]),
    {ok, _} = erllama_model:start_link(
        Model,
        model_config(?config(vision_model, Config), ?config(mmproj, Config), DiskSrv)
    ),
    [{disk_srv, DiskSrv}, {model, Model} | Config].

end_per_testcase(_TC, Config) ->
    erllama_test_helpers:stop_model_quiet(?config(model, Config)),
    erllama_test_helpers:stop_quiet(?config(disk_srv, Config)),
    ok.

model_config(Path, Mmproj, DiskSrv) ->
    #{
        backend => erllama_model_llama,
        model_path => Path,
        mmproj_path => Mmproj,
        model_opts => #{n_gpu_layers => 99},
        context_opts => #{n_ctx => 2048, n_batch => 512},
        tier_srv => DiskSrv,
        tier => disk,
        fingerprint => crypto:hash(sha256, list_to_binary(Path)),
        fingerprint_mode => safe,
        quant_type => f16,
        quant_bits => 16,
        ctx_params_hash => crypto:hash(sha256, <<"vision-suite">>),
        context_size => 2048,
        policy => #{min_tokens => 16, cold_min_tokens => 16}
    }.

%% A 32x32 24-bit BMP gradient, synthesized so the suite carries no
%% binary fixtures.
test_image() ->
    W = 32,
    H = 32,
    Row = fun(Y) ->
        Blue = min(255, Y * 8),
        Red = max(0, 255 - Y * 8),
        list_to_binary(
            lists:duplicate(W, <<Blue:8, 40:8, Red:8>>)
        )
    end,
    Pixels = <<<<(Row(Y))/binary>> || Y <- lists:seq(0, H - 1)>>,
    DataSize = byte_size(Pixels),
    FileSize = 54 + DataSize,
    Header = <<"BM", FileSize:32/little, 0:32, 54:32/little>>,
    Dib =
        <<40:32/little, W:32/little-signed, H:32/little-signed, 1:16/little, 24:16/little, 0:32,
            DataSize:32/little, 2835:32/little, 2835:32/little, 0:32, 0:32>>,
    <<Header/binary, Dib/binary, Pixels/binary>>.

image_part() ->
    #{type => image, data => test_image()}.

%% =============================================================================
%% Cases
%% =============================================================================

mmproj_caps_in_model_info(Config) ->
    Model = ?config(model, Config),
    {ok, Info} = erllama:model_info(Model),
    #{mmproj := Caps} = Info,
    ?assertEqual(true, maps:get(vision, Caps)),
    ?assert(is_boolean(maps:get(audio, Caps))),
    ok.

chat_with_image_generates(Config) ->
    Model = ?config(model, Config),
    {ok, #{message := Msg, stats := Stats}} = erllama:chat(
        Model,
        [
            #{
                role => user,
                content => [
                    #{type => text, text => <<"Describe this image briefly.">>},
                    image_part()
                ]
            }
        ],
        #{response_tokens => 32, temperature => 0.0}
    ),
    ?assert(byte_size(maps:get(content, Msg)) > 0),
    #{items := 1, n_tokens := NT, n_pos := NP} = maps:get(media, Stats),
    %% The image contributes far more positions than the short text.
    ?assert(NT > 32),
    ?assert(NP > 32),
    ok.

raw_stream_with_marker(Config) ->
    Model = ?config(model, Config),
    Prompt = <<"<|im_start|>User: <__media__>What is this?<end_of_utterance>\nAssistant:">>,
    {ok, Ref} = erllama:stream(Model, Prompt, #{
        response_tokens => 12,
        temperature => 0.0,
        media => [image_part()]
    }),
    {ok, R} = erllama:collect(Ref, 60000),
    ?assertEqual(12, length(maps:get(generated, R))),
    ?assertMatch(#{items := 1}, maps:get(media, maps:get(stats, R))),
    ok.

marker_count_mismatch_fails(Config) ->
    Model = ?config(model, Config),
    {ok, Ref} = erllama:stream(Model, <<"no marker in this prompt">>, #{
        response_tokens => 4,
        media => [image_part()]
    }),
    Result =
        receive
            {erllama, Ref, {error, Reason}} -> {error, Reason}
        after 60000 -> timeout
        end,
    ?assertMatch({error, {media_failed, _}}, Result),
    %% The model recovers for the next request.
    {ok, R} = erllama:complete(Model, <<"Say ok.">>, #{
        response_tokens => 3, temperature => 0.0
    }),
    ?assertEqual(3, length(maps:get(generated, R))),
    ok.

media_session_rejected(Config) ->
    Model = ?config(model, Config),
    ?assertEqual(
        {error, {unsupported_with_media, session_id}},
        erllama:stream(Model, <<"s <__media__>">>, #{
            media => [image_part()], session_id => sess
        })
    ),
    ok.

sequential_media_requests(Config) ->
    Model = ?config(model, Config),
    Run = fun() ->
        {ok, #{message := Msg}} = erllama:chat(
            Model,
            [
                #{
                    role => user,
                    content => [
                        #{type => text, text => <<"One word only: what is shown?">>},
                        image_part()
                    ]
                }
            ],
            #{response_tokens => 8, temperature => 0.0}
        ),
        maps:get(content, Msg)
    end,
    R1 = Run(),
    R2 = Run(),
    %% Temp-0 determinism: same image, same reply; and no cache
    %% cross-talk (media requests bypass the cache entirely).
    ?assertEqual(R1, R2),
    ok.

audio_chat_generates(Config) ->
    case {os:getenv(?AUDIO_ENV), os:getenv(?AUDIO_MMPROJ_ENV)} of
        {AM, AP} when is_list(AM), is_list(AP), AM =/= "", AP =/= "" ->
            audio_case(AM, AP, Config);
        _ ->
            {skip, "set " ?AUDIO_ENV " and " ?AUDIO_MMPROJ_ENV " to enable the audio case"}
    end.

audio_case(ModelPath, MmprojPath, Config) ->
    DiskSrv = ?config(disk_srv, Config),
    Model = <<"vision_audio_model">>,
    {ok, _} = erllama_model:start_link(
        Model, model_config(ModelPath, MmprojPath, DiskSrv)
    ),
    try
        %% 0.5 s of silence, 16 kHz mono PCM16 WAV.
        N = 8000,
        Pcm = binary:copy(<<0:16/little>>, N),
        DataSize = byte_size(Pcm),
        Wav =
            <<"RIFF", (36 + DataSize):32/little, "WAVE", "fmt ", 16:32/little, 1:16/little,
                1:16/little, 16000:32/little, 32000:32/little, 2:16/little, 16:16/little, "data",
                DataSize:32/little, Pcm/binary>>,
        {ok, #{message := Msg, stats := Stats}} = erllama:chat(
            Model,
            [
                #{
                    role => user,
                    content => [
                        #{type => text, text => <<"What do you hear?">>},
                        #{type => audio, data => Wav}
                    ]
                }
            ],
            #{response_tokens => 16, temperature => 0.0}
        ),
        ?assert(byte_size(maps:get(content, Msg)) >= 0),
        ?assertMatch(#{items := 1}, maps:get(media, Stats))
    after
        erllama_test_helpers:stop_model_quiet(Model)
    end,
    ok.
