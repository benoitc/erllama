#!/usr/bin/env python3
"""Run three local test releases with real CPU reference ranks and a small GGUF."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
from tempfile import TemporaryDirectory
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request(port, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(f"http://127.0.0.1:{port}{path}", data=data,
                  headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=15) as resp:
            return resp.status, resp.read().decode() if path == "/metrics" else json.load(resp)
    except HTTPError as exc:
        body = exc.read()
        return exc.code, json.loads(body) if body.startswith(b"{") else body


def await_status(process, port, path, status):
    deadline = time.monotonic() + 30
    while process.poll() is None and time.monotonic() < deadline:
        try:
            code, body = request(port, path)
            if code == status:
                return body
        except (URLError, TimeoutError):
            pass
        time.sleep(0.1)
    raise RuntimeError(f"{port}{path} did not return {status}, exit={process.poll()}")


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=12)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def openssl(folder, *args):
    result = subprocess.run(["openssl", *args], cwd=folder, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))


def test_ca(folder, name):
    openssl(folder, "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256",
            "-nodes", "-subj", f"/CN={name}", "-days", "1", "-sha256",
            "-addext", "basicConstraints=critical,CA:TRUE", "-addext", "keyUsage=critical,keyCertSign",
            "-keyout", f"{name}.key", "-out", f"{name}.pem")


def node_certificate(folder, ca, output):
    openssl(folder, "x509", "-req", "-in", "node.csr", "-CA", str(folder.parent / f"{ca}.pem"),
            "-CAkey", str(folder.parent / f"{ca}.key"), "-CAcreateserial", "-days", "1", "-sha256",
            "-extfile", "extensions", "-out", output)


def tls_files(folder, profile):
    shutil.copyfile(profile, folder / "tp-dist.conf")
    shutil.copyfile(folder.parent / "cluster.pem", folder / "ca.pem")
    openssl(folder, "req", "-new", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256",
            "-nodes", "-subj", f"/CN={folder.name}", "-keyout", "node.key", "-out", "node.csr")
    (folder / "extensions").write_text("basicConstraints=CA:FALSE\nkeyUsage=digitalSignature\n"
                                       "extendedKeyUsage=serverAuth,clientAuth\n"
                                       "subjectAltName=IP:127.0.0.1\n")
    node_certificate(folder, "cluster", "node.pem")


def launch(launcher, folder, env, log_name, stack):
    with (folder / log_name).open("wb") as log:
        proc = subprocess.Popen([launcher, "foreground"], cwd=folder, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    stack.callback(stop, proc)
    return proc


def await_rank(port, state):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status, body = request(port, "/rank")
        if status == 200 and body["state"] == state:
            return body
        time.sleep(0.05)
    raise RuntimeError(f"rank on port {port} did not reach {state}: {body}")


def check_rejected_peer(launcher, folder, env, stack, ports, log_name, tls_log):
    proc = launch(launcher, folder, env, log_name, stack)
    await_status(proc, ports["rank1"], "/health", 200)
    lost = await_rank(ports["rank1"], "lost")
    assert lost["reason"]["code"] == "coordinator_lost", lost
    check_group(ports["serve"], "forming")
    stop(proc)
    logs = tls_log.read_text(errors="replace").lower()
    assert "unknown ca" in logs or "unknown_ca" in logs, logs[-2000:]


def check_group(port, state, reason=None, slots=None):
    code, status = request(port, "/status")
    assert code == 200 and status["api"] == 2, status
    legacy = {"node": str, "server_vsn": str, "erllama_vsn": str, "models": list,
              "active_runs": int, "active_pulls": int, "runs": dict, "agents": list,
              "mcp": list, "cache": dict, "failures": list, "failure_counts": dict}
    assert all(isinstance(status[k], t) for k, t in legacy.items()), status
    assert {"vram", "pressure"} <= status.keys()
    group, = status["groups"]
    assert (group["group_id"], group["world"], group["generation"]) == ("smoke", 2, 1), group
    assert group["state"] == state and group["n_seq_max"] == slots, group
    why = group["reason"]
    assert (why is None if reason is None else why["code"] == reason), group
    if reason == "incompatible_ranks":
        assert why["ranks"] == [1], why
    assert [r["rank"] for r in group["ranks"]] == [0, 1]
    assert all(r["state"] == state and r["reason"] == why for r in group["ranks"]), group
    code, probe = request(port, "/ready")
    assert code == (200 if state == "ready" else 503) and probe["groups"] == [group], probe
    assert ("group" in probe.get("failed", [])) == (state != "ready"), probe
    code, metrics = request(port, "/metrics")
    labels = 'model="tiny:latest",group_id="smoke",generation="1"'
    for value in ("forming", "ready", "lost"):
        assert f'erllama_server_tp_group_state{{{labels},state="{value}"}} {int(value == state)}' in metrics
    assert f'erllama_server_tp_lost_ranks{{{labels}}} {2 if state == "lost" else 0}' in metrics


def check_rank(port, rank, state, reason=None):
    code, body = request(port, "/rank")
    assert code == 200 and (body["rank"], body["state"]) == (rank, state), body
    assert (body["reason"] is None if reason is None else body["reason"]["code"] == reason), body
    code, probe = request(port, "/ready")
    assert code == (200 if state == "ready" else 503) and probe["rank"] == body, probe
    assert ("rank" in probe.get("failed", [])) == (state != "ready"), probe


def check_refused(port, expected):
    for stream in (False, True):
        payload = {"model": "tiny", "messages": [{"role": "user", "content": "hello"}], "stream": stream}
        code, body = request(port, "/v1/chat/completions", payload)
        assert code == 503 and body["error"]["code"] == expected, (code, body)
    payload = {"model": "tiny", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1}
    code, body = request(port, "/v1/messages", payload)
    assert code == 503 and body["error"]["code"] == expected, (code, body)
    code, body = request(port, "/api/generate", {"model": "tiny", "prompt": "hello", "stream": False})
    assert code == 503 and body["code"] == expected, (code, body)


def run(root, model, tmp, stack, mismatch, tls):
    test = root / "_build/test"
    launcher = test / "rel/erllama_server/bin/erllama_server"
    drivers = test / "lib/erllama_server/test"
    if not launcher.is_file() or not (drivers / "erllama_server_cpu_rank.beam").is_file():
        raise RuntimeError("Build the test release first: rebar3 as test release")
    roles = ["serve", "rank0", "rank1"]
    if tls:
        test_ca(tmp, "cluster")
        test_ca(tmp, "untrusted")
    nodes = {r: f"tp_{r}_{os.getpid()}@127.0.0.1" for r in roles}
    processes, ports = {}, {}
    for index, role in enumerate(roles):
        folder = tmp / role
        folder.mkdir()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = ports[role] = sock.getsockname()[1]
            if tls:
                with socket.socket() as dist:
                    dist.bind(("127.0.0.1", 0))
                    dist_port = dist.getsockname()[1]
        server = f'{{data_dir,{json.dumps(str(folder / "data"))}}},{{http,#{{port=>{port}}}}}'
        path = json.dumps(str(model))
        vm = f"-name {nodes[role]}\n-kernel inet_dist_use_interface {{127,0,0,1}}\n+S 2\n"
        if tls:
            tls_files(folder, launcher.parents[1] / "etc/tp-dist.conf")
            vm += f'-proto_dist inet_tls\n-ssl_dist_optfile "{folder / "tp-dist.conf"}"\n'
            vm += f"-kernel inet_dist_listen_min {dist_port} inet_dist_listen_max {dist_port}\n"
        if role == "serve":
            group = (f'group_id=><<"smoke">>,generation=>1,world=>2,coordinator=>tp_release,'
                     f'followers=>[\'{nodes["rank1"]}\']')
            entry = f'#{{name=><<"tiny">>,path=>{path},group=>#{{{group}}}}}'
            cfg = f'[{{erllama_server,[{server},{{models,[{entry}]}}]}}].'
        else:
            ctx = 1024 if mismatch and role == "rank1" else 512
            rank = (f'group_id=><<"smoke">>,generation=>1,world=>2,rank=>{index-1},'
                    f'coordinator=>{{tp_release,\'{nodes["serve"]}\'}},model_path=>{path},'
                    'model_opts=>#{devices=>none,n_gpu_layers=>0},'
                    f'context_opts=>#{{n_ctx=>{ctx},n_seq_max=>2,n_batch=>128,n_ubatch=>128,offload_kqv=>false}}')
            cfg = (f'[{{erllama,[{{tp_rank,#{{{rank}}}}},{{tp_rank_driver,erllama_server_cpu_rank}}]}},'
                   f'{{erllama_server,[{{role,rank}},{server}]}}].')
            vm += f'-pa "{drivers}"\n'
        (folder / "sys.config").write_text(cfg + "\n")
        (folder / "vm.args").write_text(vm)
        env = dict(os.environ, RELX_CONFIG_PATH=str(folder / "sys.config"),
                   VMARGS_PATH=str(folder / "vm.args"), ERL_CRASH_DUMP=str(folder / "crash.dump"))
        env.pop("ERLLAMA_HTTP_PORT", None)
        if tls and role == "rank1":
            (folder / "node.pem").rename(folder / "trusted.pem")
            node_certificate(folder, "untrusted", "node.pem")
            check_rejected_peer(launcher, folder, env, stack, ports, "bad-client.log", tmp / "serve/boot.log")
            (folder / "trusted.pem").replace(folder / "node.pem")
            print("Serving node rejected rank 1's untrusted certificate. Group stayed forming.")
            shutil.copyfile(tmp / "untrusted.pem", folder / "ca.pem")
            check_rejected_peer(launcher, folder, env, stack, ports, "bad-server.log", folder / "bad-server.log")
            shutil.copyfile(tmp / "cluster.pem", folder / "ca.pem")
            print("Rank 1 rejected the untrusted serving certificate. Group stayed forming.")
        proc = launch(launcher, folder, env, "boot.log", stack)
        processes[role] = proc
        await_status(proc, port, "/health", 200)
        if index < 2:
            await_status(proc, port, "/ready", 503)
            if role == "serve":
                check_group(port, "forming")
            else:
                check_rank(port, 0, "forming")
    if mismatch:
        for role in roles[1:]:
            await_status(processes[role], ports[role], "/health", 200)
            await_rank(ports[role], "lost")
            check_rank(ports[role], roles.index(role) - 1, "lost", "incompatible_ranks")
        check_group(ports["serve"], "lost", "incompatible_ranks")
        check_refused(ports["serve"], "incompatible_ranks")
        print("Incompatible rank 1 reported by status, readiness, rank probes and inference.")
        return
    for role in roles:
        await_status(processes[role], ports[role], "/ready", 200)
    check_group(ports["serve"], "ready", slots=2)
    for rank, role in enumerate(roles[1:]):
        check_rank(ports[role], rank, "ready")
        assert request(ports[role], "/v1/chat/completions", {})[0] == 404
    status, body = request(ports["serve"], "/v1/chat/completions",
                           {"model": "tiny", "messages": [{"role": "user", "content": "Once upon a time"}],
                            "temperature": 0, "max_tokens": 8})
    assert status == 200, (status, body)
    answer = body["choices"][0]["message"]["content"]
    assert answer.strip(), body
    print("Three releases ready. Both CPU ranks verified the chat reply:", repr(answer))
    stop(processes["rank1"])
    await_status(processes["rank0"], ports["rank0"], "/ready", 503)
    await_status(processes["serve"], ports["serve"], "/ready", 503)
    check_group(ports["serve"], "lost", "tp_rank_lost", slots=2)
    check_rank(ports["rank0"], 0, "lost", "tp_rank_lost")
    check_refused(ports["serve"], "tp_rank_lost")
    print("Follower stopped. The serving node and surviving rank both returned 503.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--tls", action="store_true", help="test rejected and trusted rank certificates")
    args = parser.parse_args()
    for mismatch in ((False,) if args.tls else (False, True)):
        with TemporaryDirectory(prefix="erllama-release-tp-") as tmpdir:
            tmp = Path(tmpdir)
            try:
                with ExitStack() as cleanup:
                    run(Path(__file__).resolve().parents[1], args.model.resolve(strict=True), tmp, cleanup,
                        mismatch, args.tls)
            except Exception:
                for log in sorted(tmp.glob("*/*.log")):
                    print(log.parent.name, log.read_text(errors="replace")[-3500:])
                raise
