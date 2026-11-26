set -eu
cd "$(dirname "$0")"
python3 -u stage.py
export LD_LIBRARY_PATH="$HOME/erllama-tp/nccl/lib:$HOME/erllama-tp/openssl/lib"
export CUDA_VISIBLE_DEVICES=""
"$HOME/erllama-tp/otp-28.3.1/bin/erl" -noshell -pa "$HOME"/erllama-tp/erllama/_build/test/lib/*/ebin -eval '
{ok,M}=erllama_nif:load_model("vocab.gguf",#{vocab_only=>true}),{ok,T}=file:read_file("corpus.md"), Tokens=erllama_nif:tokenize(M,binary:copy(<<T/binary,"\n">>,32),#{add_special=>true,parse_special=>false}),Ids=lists:sublist(Tokens,8192),8192=length(Ids),ok=file:write_file("tokens.bin",[<<X:32/little>> || X <- Ids]),halt().'
