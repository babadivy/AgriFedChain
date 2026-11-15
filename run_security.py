"""R2-4: more poisoning scenarios x more robust aggregators, 5 seeds each."""
import json, time, numpy as np
from fl_core import *
X,y,names,paths = load_data()
SEEDS = [42,1,2,3,4]
AGGS = [("FedAvg","fedavg",{}),("Trimmed mean (b=1)","trim",{"b":1}),("Trimmed mean (b=2)","trim",{"b":2}),
        ("Coord. median","median",{}),("Krum (f=1)","krum",{"f":1})]
ATTACKS = [("No attack",None,()),("Gaussian noise, 1 client","noise",(0,)),("Sign-flip x5, 1 client","signflip",(0,)),
           ("Scaled update x10, 1 client","scale",(0,)),("Label-flip, 1 client","labelflip",(0,)),
           ("Sign-flip x5, 2 clients","signflip",(0,1))]
t0=time.time(); res={}
for seed in SEEDS:
    S = Setup(X,y,seed)
    for an,at,att in ATTACKS:
        for gn,g,kw in AGGS:
            r = run_fl(S, g, attack=at, attackers=att, **kw)
            h = r["history"]
            res.setdefault((an,gn),[]).append({"seed":seed,"final":h[-1]["test_acc"],
                "worst":min(x["test_acc"] for x in h if x["round"]>=10)})
    print("seed",seed,"done",f"{time.time()-t0:.0f}s",flush=True)
out=[]
for (an,gn),v in res.items():
    fin=np.array([x["final"] for x in v])*100; wo=np.array([x["worst"] for x in v])*100
    out.append({"attack":an,"aggregator":gn,"final_mean":fin.mean(),"final_std":fin.std(ddof=1),
                "worst_mean":wo.mean(),"worst_std":wo.std(ddof=1),"runs":v})
json.dump(out,open("security_matrix.json","w"),indent=1)
print()
print(f"{'attack':30s}|"+"|".join(f"{g[0]:>20s}" for g in AGGS))
for an,_,_ in ATTACKS:
    row=[]
    for gn,_,_ in AGGS:
        o=[x for x in out if x["attack"]==an and x["aggregator"]==gn][0]
        row.append(f"{o['final_mean']:5.1f}±{o['final_std']:3.1f} (min {o['worst_mean']:4.1f})")
    print(f"{an:30s}|"+"|".join(f"{r:>20s}" for r in row))
