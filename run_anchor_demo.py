import copy, json
from fl_core import *
X,y,names,paths = load_data(); S = Setup(X,y,42)
chain = AuditChain(); r = run_fl(S,"trim",b=1,chain=chain); ch = r["chain"]
head_before = ch.chain[-1]["hash"]                       # head hash a client/validator keeps outside the writer's control
# (a) naive edit of block 10 only
a = copy.deepcopy(ch); a.chain[10]["metrics"]["test_acc"] = 0.999
print("naive edit, internal check valid?  ", a.is_valid())
# (b) full rewrite: edit block 10 and recompute hashes of blocks 10..end (relinking prev_hash)
b = copy.deepcopy(ch); b.chain[10]["metrics"]["test_acc"] = 0.999
for i in range(10, len(b.chain)):
    if i > 10: b.chain[i]["prev_hash"] = b.chain[i-1]["hash"]
    b.chain[i]["hash"] = b._h(b.chain[i])
print("full rewrite, internal check valid? ", b.is_valid(), "(undetected by the chain alone)")
print("full rewrite vs externally kept head hash matches?", b.chain[-1]["hash"] == head_before, "(detected)")
json.dump({"naive_edit_detected": not a.is_valid()[0], "full_rewrite_internal_check_passes": b.is_valid()[0],
           "full_rewrite_detected_by_external_head_hash": b.chain[-1]["hash"] != head_before}, open("anchor_demo.json","w"), indent=1)
