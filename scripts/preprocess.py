import os
import sys
import librosa
import numpy as np

# ----------------- tweakables -----------------
TARGET_SR   = 22_050     # resample all audio here
N_MFCC      = 40         # number of MFCC coefficients
FIX_DUR_SEC = 2          # pad/trim every clip to 2 s (optional, keeps vectors equal)
DATASET_DIR = os.path.abspath(os.path.join(__file__, "..", "..", "datasets"))
FEATURE_DIR = os.path.abspath(os.path.join(__file__, "..", "..", "features"))
# ----------------------------------------------

# Create features folder if absent
os.makedirs(FEATURE_DIR, exist_ok=True)

# Try to import tqdm for progress bars; fall back if not installed.
try:
    from tqdm import tqdm
except ImportError:                         # pragma: no cover
    print("tqdm missing → pip install tqdm (progress bars will be plain).")
    def tqdm(x, **k): return x              # dummy

# ---------- 1. Unified emotion mapping ----------
EMOTION_MAP = {
    "neutral":   0,
    "calm":      1,   # only in RAVDESS
    "happy":     2,
    "sad":       3,
    "angry":     4,
    "fearful":   5,
    "disgust":   6,
    "surprised": 7
}

# ---------- 2. Filename-parsing helpers ----------
def label_from_ravdess(fn: str) -> str | None:
    """
    Example: 03-01-05-02-02-02-16.wav
             [3rd group] -> emotion code 05 → "angry"
    """
    code = int(fn.split("-")[2])
    code_to_emo = {
        1: "neutral",
        2: "calm",
        3: "happy",
        4: "sad",
        5: "angry",
        6: "fearful",
        7: "disgust",
        8: "surprised"
    }
    return code_to_emo.get(code)

def label_from_crema(fn: str) -> str | None:
    """
    Example: 1001_DFA_WANG_XX.wav
             [3rd group] -> DFA → "fearful" (DIS=disgust, HAP=happy, etc.)
    """
    code = fn.split("_")[2]
    code_to_emo = {
        "NEU": "neutral",
        "HAP": "happy",
        "SAD": "sad",
        "ANG": "angry",
        "FEA": "fearful",
        "DIS": "disgust"
    }
    return code_to_emo.get(code)

# ---------- 3. Core feature extractor ----------
def extract_mfcc(path: str) -> np.ndarray | None:
    """
    Returns a (N_MFCC,) feature vector or None on failure.
    """
    try:
        y, _ = librosa.load(path, sr=TARGET_SR, duration=FIX_DUR_SEC)
        y = librosa.util.fix_length(y, size=TARGET_SR * FIX_DUR_SEC)
        mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=N_MFCC)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:                                   # pragma: no cover
        print(f"✖ Failed {path}: {e}", file=sys.stderr)
        return None

# ---------- 4. Dataset walker ----------
def collect_dataset(root: str, parser) -> tuple[list[np.ndarray], list[int]]:
    feats, labels = [], []
    for dirpath, _, files in os.walk(root):
        wavs = [f for f in files if f.lower().endswith(".wav")]
        for wav in tqdm(wavs, desc=os.path.relpath(dirpath, DATASET_DIR)):
            emo = parser(wav)
            if emo is None or emo not in EMOTION_MAP:
                continue
            vec = extract_mfcc(os.path.join(dirpath, wav))
            if vec is not None:
                feats.append(vec)
                labels.append(EMOTION_MAP[emo])
    return feats, labels

# ---------- 5. Main runner ----------
def main() -> None:
    ravdess_root = os.path.join(DATASET_DIR, "RAVDESS")
    crema_root   = os.path.join(DATASET_DIR, "CREMA-D")

    if not os.path.isdir(ravdess_root) or not os.path.isdir(crema_root):
        sys.exit("❗  RAVDESS and CREMA-D folders not found in datasets/. Check paths.")

    X_all, y_all = [], []

    print("🔄 Processing RAVDESS …")
    X_r, y_r = collect_dataset(ravdess_root, label_from_ravdess)
    X_all += X_r; y_all += y_r

    print("🔄 Processing CREMA-D …")
    X_c, y_c = collect_dataset(crema_root, label_from_crema)
    X_all += X_c; y_all += y_c

    X_np = np.stack(X_all)
    y_np = np.array(y_all)

    print(f"✅ Combined dataset shape: {X_np.shape}, labels: {y_np.shape}")

    np.save(os.path.join(FEATURE_DIR, "X_combined.npy"), X_np)
    np.save(os.path.join(FEATURE_DIR, "y_combined.npy"), y_np)
    print("💾 Saved features/X_combined.npy and features/y_combined.npy")

if __name__ == "__main__":
    main()
