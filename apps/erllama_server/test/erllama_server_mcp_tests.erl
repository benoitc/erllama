%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_mcp_tests).
-include_lib("eunit/include/eunit.hrl").

namespace_test() ->
    ?assertEqual(<<"git__status">>, erllama_server_mcp:namespace(<<"git">>, <<"status">>)),
    %% Anything a chat template or an API could choke on becomes `_'.
    ?assertEqual(
        <<"fs__read_file_v2">>, erllama_server_mcp:namespace(<<"fs">>, <<"read.file v2">>)
    ),
    Long = erllama_server_mcp:namespace(<<"s">>, binary:copy(<<"x">>, 100)),
    ?assertEqual(64, byte_size(Long)).

render_text_and_structured_test() ->
    ?assertEqual(
        {ok, <<"one\ntwo\n{\"n\":1}">>},
        erllama_server_mcp:render(#{
            <<"content">> => [
                #{<<"type">> => <<"text">>, <<"text">> => <<"one">>},
                #{<<"type">> => <<"text">>, <<"text">> => <<"two">>}
            ],
            <<"structuredContent">> => #{<<"n">> => 1}
        })
    ).

render_error_test() ->
    ?assertEqual(
        {error, <<"bad">>},
        erllama_server_mcp:render(#{
            <<"content">> => [#{<<"type">> => <<"text">>, <<"text">> => <<"bad">>}],
            <<"isError">> => true
        })
    ).

render_media_and_resources_test() ->
    {ok, Text} = erllama_server_mcp:render(#{
        <<"content">> => [
            #{
                <<"type">> => <<"image">>,
                <<"mimeType">> => <<"image/png">>,
                <<"data">> => <<"AAAA">>
            },
            #{
                <<"type">> => <<"resource">>,
                <<"resource">> => #{<<"uri">> => <<"file:///a">>, <<"text">> => <<"body">>}
            },
            #{
                <<"type">> => <<"resource_link">>,
                <<"uri">> => <<"file:///b">>,
                <<"name">> => <<"b">>
            }
        ]
    }),
    ?assertEqual(
        <<"[image: image/png, about 3 bytes, not shown]\n[resource file:///a]\nbody\n[link file:///b b]">>,
        Text
    ).

config_test() ->
    V = fun(S) -> erllama_server_config:validate(#{mcp_servers => S}) end,
    ?assertEqual(ok, V([])),
    ?assertEqual(ok, V([#{name => "git", transport => stdio, command => "/usr/bin/git-mcp"}])),
    ?assertEqual(
        ok, V([#{name => "web", transport => http, url => "https://x/mcp", policy => auto}])
    ),
    ?assertMatch(
        {error, {invalid_config, mcp_servers, _}},
        V([#{name => "a b", transport => http, url => "https://x"}])
    ),
    ?assertMatch({error, _}, V([#{name => "a", transport => ftp}])),
    ?assertMatch(
        {error, _}, V([#{name => "a", transport => http, url => "https://x", policy => 'maybe'}])
    ),
    ?assertMatch(
        {error, _},
        V([
            #{name => "a", transport => http, url => "https://x"},
            #{name => "a", transport => http, url => "https://y"}
        ])
    ),
    ?assertMatch({error, _}, erllama_server_config:validate(#{mcp_endpoint => yes})).
