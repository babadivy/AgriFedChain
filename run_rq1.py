"""Motivation evidence: what does going federated cost, and how much of it is due to non-IID data? 5 seeds."""
import json, numpy as np
from fl_core import *
X,y,names,paths = load_data(); rows={"central":[],"fed_noniid_a0.3":[],"fed_near_iid_a100":[],"fed_noniid_a0.1":[]}
for seed in [42,1,2,3,4]:
    S = Setup(X,y,seed); rows["central"].append(run_central(S)[1][0])
    rows["fed_noniid_a0.3"].append(run_fl(S,"fedavg")["history"][-1]["test_acc"])
    for a,k in [(100,"fed_near_iid_a100"),(0.1,"fed_noniid_a0.1")]:
        Sa = Setup(X,y,seed,alpha=a)
        t = 0
        while min(len(i) for i in Sa.client_idx) < 20:   # resample degenerate partitions (empty/near-empty clients)
            t += 1; Sa.client_idx = dirichlet_partition(Sa.ytr, N_CLIENTS, a, seed + 1000 * t)
        rows[k].append(run_fl(Sa,"fedavg")["history"][-1]["test_acc"])
out={k:{"mean":float(np.mean(v)*100),"std":float(np.std(v,ddof=1)*100),"runs":[float(x) for x in v]} for k,v in rows.items()}
json.dump(out,open("rq1.json","w"),indent=1)
for k,v in out.items(): print(f"{k:20s} {v['mean']:.1f} +- {v['std']:.1f}")
