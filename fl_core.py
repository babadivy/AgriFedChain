"""AgriFedChain core (NumPy only). Same model, hyper-parameters and numerics as the submitted paper;
extended with extra aggregators, extra attacks, and a pre-training data-commitment block."""
import json, time, hashlib, copy, os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

N_CLIENTS, ALPHA = 6, 0.3
ROUNDS, LOCAL_EPOCHS, LR, BATCH, HID = 25, 2, 0.15, 32, 32
N_CLASSES = 4

# ------------------------------------------------------------------ data
def load_data(path="maize_features.npz"):
    d = np.load(path, allow_pickle=False)
    return d["X"], d["y"], [str(s) for s in d["label_names"]], d["paths"]

def dirichlet_partition(y, n_clients, alpha, seed):
    r = np.random.default_rng(seed)
    ci = [[] for _ in range(n_clients)]
    for c in range(y.max() + 1):
        idx = np.where(y == c)[0]; r.shuffle(idx)
        prop = r.dirichlet(alpha * np.ones(n_clients))
        cuts = (np.cumsum(prop) * len(idx)).astype(int)[:-1]
        for i, p in enumerate(np.split(idx, cuts)):
            ci[i].extend(p.tolist())
    return [np.array(sorted(c), dtype=int) for c in ci]

class Setup:
    def __init__(self, X, y, seed, n_clients=N_CLIENTS, alpha=ALPHA):
        self.seed = seed
        self.idx_train, self.idx_test = train_test_split(
            np.arange(len(y)), test_size=0.20, random_state=seed, stratify=y)
        sc = StandardScaler().fit(X[self.idx_train])
        self.Xtr = sc.transform(X[self.idx_train]).astype(np.float32)
        self.Xte = sc.transform(X[self.idx_test]).astype(np.float32)
        self.ytr, self.yte = y[self.idx_train], y[self.idx_test]
        self.client_idx = dirichlet_partition(self.ytr, n_clients, alpha, seed)

# ------------------------------------------------------------------ model
def init_params(seed, in_dim=54):
    r = np.random.default_rng(seed)
    l1 = np.sqrt(6 / (in_dim + HID)); l2 = np.sqrt(6 / (HID + N_CLASSES))
    return {"W1": r.uniform(-l1, l1, (in_dim, HID)).astype(np.float32),
            "b1": np.zeros(HID, dtype=np.float32),
            "W2": r.uniform(-l2, l2, (HID, N_CLASSES)).astype(np.float32),
            "b2": np.zeros(N_CLASSES, dtype=np.float32)}

def forward(p, X):
    z1 = X @ p["W1"] + p["b1"]; a1 = np.maximum(0, z1)
    z2 = a1 @ p["W2"] + p["b2"]; z2 = z2 - z2.max(axis=1, keepdims=True)
    e = np.exp(z2); return z1, a1, e / e.sum(axis=1, keepdims=True)

def loss_and_grad(p, X, yoh):
    n = X.shape[0]; z1, a1, pr = forward(p, X)
    dz2 = (pr - yoh) / n
    da1 = dz2 @ p["W2"].T; dz1 = da1 * (z1 > 0)
    return {"W1": X.T @ dz1, "b1": dz1.sum(0), "W2": a1.T @ dz2, "b2": dz2.sum(0)}

def one_hot(y, k=N_CLASSES):
    o = np.zeros((len(y), k), dtype=np.float32); o[np.arange(len(y)), y] = 1.0; return o

def local_train(params, X, y, epochs, lr, batch, seed):
    r = np.random.default_rng(seed); p = copy.deepcopy(params); n = X.shape[0]; yoh = one_hot(y)
    for _ in range(epochs):
        perm = r.permutation(n)
        for s in range(0, n, batch):
            b = perm[s:s + batch]; g = loss_and_grad(p, X[b], yoh[b])
            for k in p: p[k] -= lr * g[k]
    return p

def evaluate(p, X, y):
    _, _, pr = forward(p, X); pred = pr.argmax(1)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return accuracy_score(y, pred), prec, rec, f1, pred

def flatten(p): return np.concatenate([p[k].ravel() for k in ["W1", "b1", "W2", "b2"]])
def param_bytes(p): return sum(v.nbytes for v in p.values())
PARAM_COUNT = 54 * HID + HID + HID * N_CLASSES + N_CLASSES  # 1892

# ------------------------------------------------------------------ aggregators
def fedavg_weighted(plist, sizes):
    w = np.array(sizes, dtype=np.float64); w = w / w.sum()
    return {k: sum(wi * p[k] for wi, p in zip(w, plist)) for k in plist[0]}

def aggregate(plist, sizes, agg, b=1, f=1):
    if agg == "fedavg": return fedavg_weighted(plist, sizes)
    out = {}
    if agg in ("trim", "median"):
        for k in plist[0]:
            st = np.stack([p[k] for p in plist], axis=0)
            if agg == "trim": out[k] = np.sort(st, axis=0)[b:len(plist) - b].mean(axis=0)
            else: out[k] = np.median(st, axis=0)
        return out
    if agg == "krum":  # Krum: pick the update closest to its K-f-2 nearest neighbours
        V = np.stack([flatten(p) for p in plist]).astype(np.float64); K = len(V)
        d2 = ((V[:, None, :] - V[None, :, :]) ** 2).sum(-1); m = K - f - 2
        scores = [np.sort(np.delete(d2[i], i))[:m].sum() for i in range(K)]
        return copy.deepcopy(plist[int(np.argmin(scores))])
    raise ValueError(agg)

# ------------------------------------------------------------------ hashing / ledger
def sha_arr(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""): h.update(chunk)
    return h.hexdigest()

def merkle_root(leaves):
    if not leaves: return hashlib.sha256(b"").hexdigest()
    lvl = list(leaves)
    while len(lvl) > 1:
        if len(lvl) % 2: lvl.append(lvl[-1])
        lvl = [hashlib.sha256((lvl[i] + lvl[i + 1]).encode()).hexdigest() for i in range(0, len(lvl), 2)]
    return lvl[0]

class AuditChain:
    """Hash-chained, single-writer ledger. Block 0 = genesis; block 1 (optional) = pre-training data commitment;
    then one block per training round."""
    def __init__(self):
        g = {"index": 0, "prev_hash": "0" * 64, "type": "genesis", "round": -1}
        g["hash"] = self._h(g); self.chain = [g]
    @staticmethod
    def _h(b): return hashlib.sha256(json.dumps({k: v for k, v in b.items() if k != "hash"}, sort_keys=True).encode()).hexdigest()
    def _add(self, body):
        b = {"index": len(self.chain), "prev_hash": self.chain[-1]["hash"], **body}
        b["hash"] = self._h(b); self.chain.append(b); return b
    def commit_data(self, client_roots):
        return self._add({"type": "data_commitment", "round": 0, "client_data_roots": client_roots})
    def add_round(self, rnd, gparams, upd_hashes, metrics):
        return self._add({"type": "round", "round": rnd, "global_model_hash": sha_arr(flatten(gparams)),
                          "client_update_hashes": upd_hashes, "metrics": metrics})
    def is_valid(self):
        for i in range(1, len(self.chain)):
            b, pv = self.chain[i], self.chain[i - 1]
            if b["prev_hash"] != pv["hash"] or self._h(b) != b["hash"]: return False, i
        return True, None

def client_data_roots(S, paths, file_hash_cache):
    """Merkle root over SHA-256 of every raw image file in each client's shard."""
    ptr = paths[S.idx_train]; roots = []
    for idx in S.client_idx:
        files = sorted(str(p) for p in ptr[idx])
        roots.append(merkle_root([file_hash_cache[f] for f in files]))
    return roots

# ------------------------------------------------------------------ one FL run
def run_fl(S, agg="fedavg", b=1, f=1, attack=None, attackers=(), attack_from=10, rounds=ROUNDS,
           chain=None, gamma_flip=5.0, gamma_scale=10.0, noise_sigma=3.0):
    seed = S.seed; params = init_params(seed, S.Xtr.shape[1]); hist = []; comm_actual = 0
    for rnd in range(1, rounds + 1):
        local, sizes, hashes = [], [], []
        for cid, idx in enumerate(S.client_idx):
            Xc, yc = S.Xtr[idx], S.ytr[idx]; ls = seed * 1000 + rnd * 10 + cid
            mal = attack is not None and cid in attackers and rnd >= attack_from
            if mal and attack == "noise":
                r = np.random.default_rng(1000 + rnd + 100000 * cid)
                pc = {k: v + r.normal(0, noise_sigma, v.shape).astype(np.float32) for k, v in params.items()}
            elif mal and attack in ("signflip", "scale"):
                ph = local_train(params, Xc, yc, LOCAL_EPOCHS, LR, BATCH, ls)
                g = -gamma_flip if attack == "signflip" else gamma_scale
                pc = {k: params[k] + g * (ph[k] - params[k]) for k in params}
            elif mal and attack == "labelflip":
                pc = local_train(params, Xc, (yc + 1) % N_CLASSES, LOCAL_EPOCHS, LR, BATCH, ls)
            else:
                pc = local_train(params, Xc, yc, LOCAL_EPOCHS, LR, BATCH, ls)
            local.append(pc); sizes.append(len(idx)); hashes.append(sha_arr(flatten(pc)))
            comm_actual += 2 * param_bytes(pc)
        params = aggregate(local, sizes, agg, b=b, f=f)
        acc, prec, rec, f1, _ = evaluate(params, S.Xte, S.yte)
        m = {"test_acc": round(float(acc), 4), "test_f1_macro": round(float(f1), 4)}
        if chain is not None: chain.add_round(rnd, params, hashes, m)
        hist.append({"round": rnd, **m})
    return {"params": params, "history": hist, "chain": chain, "comm_bytes_actual": comm_actual}

def run_central(S):
    p = local_train(init_params(S.seed, S.Xtr.shape[1]), S.Xtr, S.ytr, ROUNDS * LOCAL_EPOCHS, LR, BATCH, S.seed)
    return p, evaluate(p, S.Xte, S.yte)
