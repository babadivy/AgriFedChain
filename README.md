# AgriFedChain - experiment code

Requirements: Python 3.10+, numpy, scikit-learn, scikit-image, Pillow, matplotlib.

Data: the four maize classes of PlantVillage
(github.com/spMohanty/PlantVillage-Dataset, raw/color/Corn_(maize)___*), placed in pv/raw/color/.
The extracted features (maize_features.npz) are included, so steps 2-9 run without the images,
except run_commit.py and run_ledger_final.py, which hash the raw image files.

1. python3 extract_features.py pv/raw/color maize_features.npz   # 54-dim features
2. python3 repro_check.py       # seed 42: FedAvg 0.9494, centralized 0.9585, attack minimum 0.2672, trimmed mean 0.917
3. python3 make_figures.py      # Figs. 2-4 and per-class results (Tables II, III)
4. python3 run_rq1.py           # centralized vs. FedAvg over five seeds (Section V-A)
5. python3 run_security.py      # Table IV: aggregators x attacks x five seeds
6. python3 run_commit.py        # data commitment before training and tamper test
7. python3 run_ledger_final.py  # ledger tests incl. full rewrite and external head hash (Section V-D)
8. python3 run_scalability.py   # aggregation time, traffic, ledger size vs. number of clients
9. python3 run_parallel.py --workers 1 2 3 6 --reps 3   # emulated multi-node execution (Section V-F)

fl_core.py holds the model, local training, aggregation rules (FedAvg, trimmed mean, median, Krum),
attacks, the hash-chained ledger and the Merkle data commitment.
results/ contains the JSON files with the numbers reported in the paper.
