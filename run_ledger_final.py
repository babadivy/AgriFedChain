"""Ledger tests on the FINAL design: genesis, block 1 = data commitment, blocks 2..26 = rounds 1..25."""
import copy, json, os, shutil, tempfile
from fl_core import *
X,y,names,paths = load_data(); S = Setup(X,y,42)
cache = {str(p): sha_file(str(p)) for p in paths}
roots = client_data_roots(S, paths, cache)
chain = AuditChain(); chain.commit_data(roots); r = run_fl(S,"trim",b=1,chain=chain); ch = r["chain"]
out = {"blocks": len(ch.chain), "valid": ch.is_valid()[0], "round10_block_index": next(b["index"] for b in ch.chain if b.get("round")==10 and b["type"]=="round")}
idx10 = out["round10_block_index"]; head = ch.chain[-1]["hash"]
# (a) naive edit of round-10 block
a = copy.deepcopy(ch); a.chain[idx10]["metrics"]["test_acc"] = 0.999; v = a.is_valid(); out["naive_edit_detected"] = (not v[0]); out["naive_edit_break_index"] = v[1]
# (b) full rewrite: edit and recompute every later hash
b = copy.deepcopy(ch); b.chain[idx10]["metrics"]["test_acc"] = 0.999
for i in range(idx10, len(b.chain)):
    if i > idx10: b.chain[i]["prev_hash"] = b.chain[i-1]["hash"]
    b.chain[i]["hash"] = b._h(b.chain[i])
out["full_rewrite_passes_internal_check"] = b.is_valid()[0]; out["full_rewrite_detected_with_external_head"] = (b.chain[-1]["hash"] != head)
# (c) edit the data-commitment block
c = copy.deepcopy(ch); c.chain[1]["client_data_roots"][3] = "0"*64; v = c.is_valid(); out["commitment_edit_detected"] = (not v[0]); out["commitment_edit_break_index"] = v[1]
# (d) one-bit change in one image of client 3
ptr = paths[S.idx_train]; victim = str(ptr[S.client_idx[3][0]]); tmp = tempfile.mkdtemp(); cp = os.path.join(tmp,"img.jpg"); shutil.copy(victim,cp)
bb = bytearray(open(cp,"rb").read()); bb[len(bb)//2] ^= 0x01; open(cp,"wb").write(bytes(bb))
c2 = dict(cache); c2[victim] = sha_file(cp); after = [x==y_ for x,y_ in zip(client_data_roots(S,paths,c2), roots)]
out["onebit_tamper_root_matches"] = after; out["hashing_seconds_3852_files"] = None
import time; t=time.time(); [sha_file(str(p)) for p in paths]; out["hashing_seconds_3852_files"] = round(time.time()-t,2)
json.dump(out, open("ledger_final.json","w"), indent=1); print(json.dumps(out, indent=1))
