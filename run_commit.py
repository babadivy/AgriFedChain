"""R1: ledger used BEFORE training -> each client commits a Merkle root over its raw image files in block 1."""
import json, os, shutil, tempfile, time, numpy as np
from fl_core import *
X,y,names,paths = load_data()
S = Setup(X,y,42)
t=time.time(); cache = {str(p): sha_file(str(p)) for p in paths}; t_hash=time.time()-t
roots = client_data_roots(S, paths, cache)
chain = AuditChain(); chain.commit_data(roots)          # block 1 = pre-training commitment
r = run_fl(S, "trim", b=1, chain=chain)                  # blocks 2.. = training rounds
print(f"hashed {len(cache)} image files in {t_hash:.1f}s; chain length {len(chain.chain)} blocks; valid={chain.is_valid()}")
# (a) shards unchanged -> every root still matches the on-ledger commitment
same = [a==b for a,b in zip(client_data_roots(S, paths, cache), chain.chain[1]["client_data_roots"])]
print("shard roots match commitment:", same)
# (b) tamper with ONE real image of client 3 (flip one byte in a copy) and re-verify
ptr = paths[S.idx_train]; victim = str(ptr[S.client_idx[3][0]])
tmp = tempfile.mkdtemp(); cp = os.path.join(tmp, "img.jpg"); shutil.copy(victim, cp)
b = bytearray(open(cp,"rb").read()); b[len(b)//2] ^= 0x01; open(cp,"wb").write(bytes(b))
cache2 = dict(cache); cache2[victim] = sha_file(cp)
after = [a==b for a,b in zip(client_data_roots(S, paths, cache2), chain.chain[1]["client_data_roots"])]
print("after flipping 1 byte in 1 image of client 3 ->", after)
# (c) rewrite the commitment on the ledger instead -> chain breaks
import copy; ch2 = copy.deepcopy(chain); ch2.chain[1]["client_data_roots"][3] = "0"*64
print("edit commitment block on ledger -> valid, break_at:", ch2.is_valid())
json.dump({"n_files":len(cache),"hash_seconds":round(t_hash,2),"roots":roots,"unchanged_match":same,
           "one_byte_tamper_match":after,"ledger_edit_detected":(not ch2.is_valid()[0]),
           "chain_blocks":len(chain.chain),"chain_valid":chain.is_valid()[0]}, open("commit_demo.json","w"), indent=1)
