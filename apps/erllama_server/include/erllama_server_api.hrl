%% The service API version. The release and the CLI escript are built
%% from the same tree; the CLI sends this in `hello/1' and the release
%% refuses any other value. Bump it on an incompatible change to a
%% function or event shape of erllama_server_api.
-define(ERLLAMA_SERVER_API_VSN, 1).
