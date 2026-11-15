
import os, sys, time, numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import local_binary_pattern
from skimage.filters import sobel

ROOT = sys.argv[1] if len(sys.argv) > 1 else "pv/raw/color"
OUT = sys.argv[2] if len(sys.argv) > 2 else "maize_features.npz"
CLASSES = [
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
]
LABEL_NAMES = ["Gray_Leaf_Spot", "Common_Rust", "Northern_Leaf_Blight", "Healthy"]
IMG_SIZE, LBP_P, LBP_R = 64, 8, 1

def extract_one(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img).astype(np.float64) / 255.0
    hsv = rgb2hsv(arr)
    feats = []
    for c in range(3):
        hist, _ = np.histogram(hsv[:, :, c], bins=12, range=(0, 1))
        feats.extend((hist / hist.sum()).tolist())
    for c in range(3):
        feats.append(arr[:, :, c].mean()); feats.append(arr[:, :, c].std())
    gray = rgb2gray(arr)
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method="uniform")
    h, _ = np.histogram(lbp, bins=LBP_P + 2, range=(0, LBP_P + 2))
    feats.extend((h / h.sum()).tolist())
    e = sobel(gray)
    feats.append(e.mean()); feats.append(e.std())
    return np.array(feats, dtype=np.float32)

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    t0 = time.time(); X, y, paths = [], [], []
    for li, cls in enumerate(CLASSES):
        d = os.path.join(ROOT, cls)
        for fn in sorted(os.listdir(d)):
            fp = os.path.join(d, fn)
            X.append(extract_one(fp)); y.append(li); paths.append(fp)
    X = np.stack(X); y = np.array(y, dtype=np.int64)
    np.savez_compressed(OUT, X=X, y=y, label_names=np.array(LABEL_NAMES), paths=np.array(paths))
    print("features", X.shape, "labels", y.shape, f"{time.time()-t0:.1f}s ->", OUT)
