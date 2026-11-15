"""R2-2: how do the aggregator, the ledger and the traffic scale with the number of clients K (model size d = 1,892)?
Measured on whatever machine runs it (single core in our sandbox)."""
import json, time, hashlib, numpy as np
from fl_core import PARAM_COUNT, AuditChain

d = PARAM_COUNT; rng = np.random.default_rng(0)

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)

def krum_fast(U, f):
    V = U.astype(np.float64); sq = (V ** 2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2 * V @ V.T; np.fill_diagonal(d2, np.inf)
    m = len(V) - f - 2
    scores = np.partition(d2, m - 1, axis=1)[:, :m].sum(1)
    return V[int(np.argmin(scores))]

rows = []
for K in [6, 60, 600, 2000, 6000]:
    U = rng.standard_normal((K, d)).astype(np.float32); w = np.ones(K); b = max(1, K // 10); f = max(1, K // 10)
    r = {"K": K, "b": b,
         "fedavg_ms": best(lambda: np.average(U, axis=0, weights=w)) * 1e3,
         "trimmed_ms": best(lambda: np.sort(U, axis=0)[b:K - b].mean(0)) * 1e3,
         "median_ms": best(lambda: np.median(U, axis=0)) * 1e3,
         "krum_ms": (best(lambda: krum_fast(U, f), reps=2) * 1e3) if K <= 2000 else None,
         "traffic_MB_per_round_float32": 2 * K * d * 4 / 1e6}
    # ledger: one block with K per-client update hashes
    hs = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(K)]
    ch = AuditChain()
    t = time.perf_counter()
    for rnd in range(1, 26): ch._add({"type": "round", "round": rnd, "global_model_hash": "0" * 64, "client_update_hashes": hs, "metrics": {}})
    r["ledger_append_ms_per_block"] = (time.perf_counter() - t) / 25 * 1e3
    r["ledger_block_KB"] = len(json.dumps(ch.chain[-1])) / 1e3
    t = time.perf_counter(); ok = ch.is_valid()[0]; r["ledger_verify_25_blocks_ms"] = (time.perf_counter() - t) * 1e3
    rows.append(r); print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(rows, open("scalability.json", "w"), indent=1)
