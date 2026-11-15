"""R1: emulate multi-node execution on one machine: each worker is a separate OS process owning a subset of the
6 clients; every download/upload is delayed by an emulated link (RTT + bytes/bandwidth). Reports round wall-time.
Usage: python3 run_parallel.py [--workers 1 2 3 6] [--reps 3]   (run on a multi-core machine for real compute speedup)"""
import argparse, json, os, time, statistics, multiprocessing as mp
import numpy as np
from fl_core import *

PROFILES = {"No network delay": (None, 0.0), "Broadband (100 Mbps, 10 ms RTT)": (100, 10),
            "4G (10 Mbps, 50 ms RTT)": (10, 50), "Rural 3G (1 Mbps, 150 ms RTT)": (1, 150)}
PAYLOAD = PARAM_COUNT * 4            # float32 model bytes

def delay(profile, extra=0):
    bw, rtt = PROFILES[profile]
    return 0.0 if bw is None else rtt / 2000.0 + (PAYLOAD + extra) * 8 / (bw * 1e6)

def worker(conn, ids, S, d_down, d_up):
    while True:
        msg = conn.recv()
        if msg is None: break
        rnd, params = msg; out = {}
        for cid in ids:
            time.sleep(d_down)
            idx = S.client_idx[cid]
            out[cid] = local_train(params, S.Xtr[idx], S.ytr[idx], LOCAL_EPOCHS, LR, BATCH, S.seed * 1000 + rnd * 10 + cid)
            time.sleep(d_up)
        conn.send(out)

def run(S, W, profile):
    ctx = mp.get_context("fork"); pipes, procs = [], []
    dn, up = delay(profile), delay(profile, extra=32)
    for w in range(W):
        pc, cc = ctx.Pipe(); ids = [c for c in range(N_CLIENTS) if c % W == w]
        p = ctx.Process(target=worker, args=(cc, ids, S, dn, up)); p.start(); pipes.append(pc); procs.append(p)
    params = init_params(S.seed, S.Xtr.shape[1]); sizes = [len(i) for i in S.client_idx]
    t0 = time.perf_counter()
    for rnd in range(1, ROUNDS + 1):
        for pc in pipes: pc.send((rnd, params))
        got = {}
        for pc in pipes: got.update(pc.recv())
        params = fedavg_weighted([got[c] for c in range(N_CLIENTS)], sizes)
    T = time.perf_counter() - t0
    for pc in pipes: pc.send(None)
    for p in procs: p.join()
    return T, evaluate(params, S.Xte, S.yte)[0]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 3, 6])
    ap.add_argument("--reps", type=int, default=3); a = ap.parse_args()
    X, y, names, paths = load_data(); S = Setup(X, y, 42)
    ref = evaluate(run_fl(S, "fedavg")["params"], S.Xte, S.yte)[0]
    cores = len(os.sched_getaffinity(0)); print("usable CPU cores:", cores, "| sequential reference acc:", round(ref, 4))
    rows = []
    for prof in PROFILES:
        reps = a.reps if PROFILES[prof][0] is None else 1
        base = None
        for W in a.workers:
            ts = []; acc = None
            for _ in range(reps):
                T, acc = run(S, W, prof); ts.append(T)
            T = statistics.median(ts); base = base or T
            rows.append({"profile": prof, "workers": W, "total_s": round(T, 3), "per_round_s": round(T / ROUNDS, 4),
                         "speedup_vs_1_worker": round(base / T, 2), "final_acc": round(acc, 4), "acc_matches_sequential": abs(acc - ref) < 1e-9})
            print(rows[-1], flush=True)
    json.dump({"usable_cores": cores, "payload_bytes_float32": PAYLOAD, "rows": rows}, open("parallel_emulation.json", "w"), indent=1)
