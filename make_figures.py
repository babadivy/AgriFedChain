import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from fl_core import *
plt.rcParams.update({"font.size":9,"axes.labelsize":9,"legend.fontsize":8,"figure.dpi":220})
X,y,names,paths = load_data(); S = Setup(X,y,42)
fed = run_fl(S,"fedavg"); cp,(ca,_,_,_,_) = run_central(S)
pl = run_fl(S,"fedavg",attack="noise",attackers=(0,)); tr = run_fl(S,"trim",b=1,attack="noise",attackers=(0,))
r = list(range(1,ROUNDS+1))
fig,ax = plt.subplots(figsize=(3.4,2.3)); ax.plot(r,[h["test_acc"] for h in fed["history"]],marker="o",ms=3,lw=1.3,color="#1b6ca8",label="Federated (FedAvg, non-IID, K=6)")
ax.axhline(ca,color="#c1440e",ls="--",lw=1.2,label=f"Centralized ({ca*100:.1f}%)"); ax.set_xlabel("Communication round"); ax.set_ylabel("Test accuracy"); ax.set_ylim(0.8,1.0); ax.legend(loc="lower right",frameon=False); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig("fig_convergence.png",bbox_inches="tight"); plt.close(fig)
a,pm,rm,fm,pred = evaluate(fed["params"],S.Xte,S.yte); cm = confusion_matrix(S.yte,pred); short=["GLS","Rust","NLB","Healthy"]
fig,ax = plt.subplots(figsize=(2.9,2.4)); im=ax.imshow(cm,cmap="Blues"); ax.set_xticks(range(4)); ax.set_xticklabels(short,rotation=30,ha="right"); ax.set_yticks(range(4)); ax.set_yticklabels(short)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
for i in range(4):
    for j in range(4): ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=8,color="white" if cm[i,j]>cm.max()/2 else "black")
fig.colorbar(im,ax=ax,fraction=.046,pad=.04); fig.tight_layout(); fig.savefig("fig_confusion.png",bbox_inches="tight"); plt.close(fig)
fig,ax = plt.subplots(figsize=(3.4,2.3)); ax.plot(r,[h["test_acc"] for h in pl["history"]],marker="s",ms=3,lw=1.3,color="#c1440e",label="FedAvg (plain averaging)")
ax.plot(r,[h["test_acc"] for h in tr["history"]],marker="o",ms=3,lw=1.3,color="#1b6ca8",label="Trimmed mean (b=1)"); ax.axvline(10,color="gray",ls=":",lw=1.2,label="Attack starts (round 10)")
ax.set_xlabel("Communication round"); ax.set_ylabel("Test accuracy"); ax.set_ylim(0,1); ax.legend(loc="lower left",frameon=False,fontsize=6.6); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig("fig_byzantine.png",bbox_inches="tight"); plt.close(fig)
p,rc,f,sup = precision_recall_fscore_support(S.yte,pred,zero_division=0)
json.dump({"cm":cm.tolist(),"per_class":{n:[round(float(p[i]),3),round(float(rc[i]),3),round(float(f[i]),3),int(sup[i])] for i,n in enumerate(names)},
 "fed":[round(float(a),4),round(float(pm),4),round(float(rm),4),round(float(fm),4)],"central_acc":round(float(ca),4)},open("seed42_detail.json","w"),indent=1)
print("fed",a,pm,rm,fm,"central",ca); print(cm.tolist())
for i,n in enumerate(names): print(n,round(float(p[i]),3),round(float(rc[i]),3),round(float(f[i]),3),int(sup[i]))
