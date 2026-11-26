import argparse
import glob
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT = Path.cwd()
REPO = Path(__file__).resolve().parents[3]
PORT = 19083
NAME = 'qwen-27b'
PROMPTS = [
    'Navigation at sea changed with the arrival of instruments for measuring time, angles and distance. Write a detailed chronological explanation for a student. Include examples from at least four periods and explain how a navigator checked a position for errors.',
    'Erlang applications often supervise workers that exchange messages. Write a detailed technical guide for a developer implementing a bounded queue. Explain admission, back pressure, worker recovery and how to test concurrent requests. Include an example.',
    'Bridge designers must account for wind, changing loads and temperature. Explain the engineering of a long suspension bridge in five detailed paragraphs. Give a concrete example of each design trade-off and describe how engineers check their assumptions.',
]
ERL = """
application:load(erllama),
application:set_env(erllama,ram_tier_budget,1073741824),
application:set_env(erllama,min_tokens,32),
application:set_env(erllama,cold_min_tokens,32),
application:set_env(erllama,boundary_trim_tokens,1),
application:set_env(erllama,boundary_align_tokens,1),
application:load(erllama_server),
application:set_env(erllama_server,data_dir,os:getenv("MAC_BENCH_DATA")),
application:set_env(erllama_server,http,#{ip=>{127,0,0,1},port=>19083}),
{ok,_}=application:ensure_all_started(erllama_server),
io:format("DEVICES ~p~n",[erllama:list_devices()]),
{ok,_}=erllama_server_api:load_model(os:getenv("MAC_BENCH_MODEL"),#{name=><<"qwen-27b">>,n_ctx=>2048,n_gpu_layers=>999,parallel=>1,keep_alive=>infinity}),
{module,erllama_server_generate}=code:ensure_loaded(erllama_server_generate),
Tracer=spawn(fun Loop() -> receive {trace,_,call,{erllama_server_generate,finish,4},Stats} -> io:format("CACHE_STATS ~s~n",[json:encode(maps:with([cache_hit_kind,cache_delta,prompt_tokens,completion_tokens,prefill_ms],Stats))]), Loop(); _ -> Loop() end end),
erlang:trace_pattern({erllama_server_generate,finish,4},[{['_','_','_','$1'],[],[{message,'$1'}]}],[]),
erlang:trace(all,true,[call,arity,{tracer,Tracer}]),
io:format("BENCH_READY~n"), receive stop_bench -> ok end.
"""

def request(engine, prompt, cap):
    body={'model': NAME, 'messages':[{'role':'user','content':prompt}], 'temperature':0, 'max_tokens':cap, 'stream': True, 'stream_options':{'include_usage':True}}
    if engine=='erllama':
        body['reasoning_effort']='none'
    else:
        body['chat_template_kwargs']={'enable_thinking':False}
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions', json.dumps(body).encode(), {'Content-Type':'application/json'})
    start=time.monotonic()
    first=None
    last=None
    parts=[]
    reasoning=[]
    finish=None
    usage={}
    with urllib.request.urlopen(req,timeout=180) as r:
        for line in r:
            if not line.startswith(b'data:'): continue
            payload=line[5:].strip()
            if payload==b'[DONE]': break
            ev=json.loads(payload)
            if 'error' in ev: raise ValueError(ev['error'])
            usage=ev.get('usage') or usage
            for ch in ev.get('choices',[]):
                delta=ch.get('delta',{})
                if delta.get('content'):
                    last=time.monotonic()
                    if first is None: first=last
                    parts.append(delta['content'])
                if delta.get('reasoning_content'): reasoning.append(delta['reasoning_content'])
                finish=ch.get('finish_reason') or finish
    end=time.monotonic()
    assert first and usage and not reasoning,(usage,finish,reasoning[:1])
    rows=dict(prompt=prompt,ttft_ms=1000*(first-start),wall_s=end-start,tokens=usage['completion_tokens'],usage=usage,
              finish=finish,output=''.join(parts),visible_chunks=len(parts),reasoning_bytes=0,last_text_s=last-start)
    rows['e2e_tps']=rows['tokens']/rows['wall_s']
    if last>first: rows['visible_decode_tps_est']=(rows['tokens']-1)/(last-first)
    return rows

def run(engine, repeat=1):
    prefix=engine if repeat==1 else engine+'-reverse'
    log=ROOT/(prefix+'-serve.log')
    outfile=ROOT/(prefix+'-results.json')
    env=dict(os.environ,ERL_CRASH_DUMP=str(ROOT/'erl_crash.dump'),MAC_BENCH_DATA=str(ROOT/('server-data-'+str(repeat))))
    if engine=='erllama':
        env['MAC_BENCH_MODEL']=str(next((ROOT/'gguf').glob('*.gguf')))
        args=['erl','+S','8:8','-noshell','-pa',*glob.glob(str(REPO/'_build/default/lib/*/ebin')),'-eval',ERL]
    else:
        args=[str(ROOT/'tensorfold-1.3.5-macos-arm64/bin/tensorfold-native'),'serve',str(ROOT/'mlx'),'--name',NAME,'--host','127.0.0.1','--port',str(PORT),'--context','2048','--parallel','1','--no-drafts','--no-thinking','--prompt-cache-gib','1','--no-update-check']
    with log.open('w') as f:
        proc=subprocess.Popen(args,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            for i in range(600):
                if proc.poll() is not None: raise RuntimeError(f'{engine} exited {proc.returncode}, see {log}')
                if engine=='erllama' and 'BENCH_READY' not in log.read_text(errors='replace'): time.sleep(.5);continue
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models',timeout=2) as resp:
                        models=json.load(resp)
                    if any(x['id'] in (NAME, NAME + ':latest') for x in models['data']): break
                except OSError: pass
                time.sleep(.5)
            else: raise RuntimeError(f'{engine} startup timed out')
            request(engine,'Write a short sentence about rain.',8)
            rows=[]
            order=range(len(PROMPTS)) if repeat==1 else reversed(range(len(PROMPTS)))
            for i in order:
                prompt=PROMPTS[i]
                for phase in ['first','retry']:
                    rec=request(engine,prompt,128)
                    assert rec['tokens']==128,(rec['tokens'],rec['finish'])
                    rows.append(dict(engine=engine,case=i+1,phase=phase,**rec))
                    outfile.write_text(json.dumps(rows,indent=2))
                    print(engine,i+1,phase,round(rec['ttft_ms']),round(rec['e2e_tps'],2),flush=True)
                    time.sleep(.5)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait()

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('engine',choices=['erllama','tensorfold'])
    ap.add_argument('--pass-number', type=int, choices=[1,2], default=1)
    args=ap.parse_args()
    run(args.engine,args.pass_number)
