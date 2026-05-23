import os
import warnings
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")

STAGE_MAP = {
    "Sleep stage W": 0,    
    "Sleep stage 1": 1,    
    "Sleep stage 2": 2,    
    "Sleep stage 3": 3,    
    "Sleep stage 4": 3,    
    "Sleep stage R": 4,    
    "Sleep stage ?": -1,   
    "Movement time": -1,   
}

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

EPOCH_SEC = 30   
FS = 100         

def band_power(data, fs, fmin, fmax):
    freqs, psd = welch(data, fs=fs, nperseg=min(256, len(data)))
    mask = (freqs >= fmin) & (freqs <= fmax)

    return float(np.mean(psd[mask])) if mask.any() else 0.0


def compute_features(epoch_eeg, epoch_eog, epoch_emg, fs):
    delta = band_power(epoch_eeg, fs, 0.5, 4.0)
    theta = band_power(epoch_eeg, fs, 4.0, 8.0)
    alpha = band_power(epoch_eeg, fs, 8.0, 13.0)
    beta = band_power(epoch_eeg, fs, 13.0, 30.0)
    sigma = band_power(epoch_eeg, fs, 11.0, 16.0)
    total = delta + theta + alpha + beta + 1e-12  
    spindle_ratio = sigma / total
    eog_var = float(np.var(epoch_eog)) if epoch_eog is not None else 0.0
    emg_var = float(np.var(epoch_emg)) if epoch_emg is not None else 0.0
    
    hist, _ = np.histogram(epoch_eeg, bins=50, density=True)
    sig_entropy = float(scipy_entropy(hist + 1e-12))
    
    return {
        "delta_power": delta,
        "theta_power": theta,
        "alpha_power": alpha,
        "beta_power": beta,
        "spindle_ratio": spindle_ratio,
        "eog_variance": eog_var,
        "emg_variance": emg_var,
        "signal_entropy": sig_entropy,
    }


def load_and_extract(data_dir="data", subject_idx=0, save_csv=True):
    print(f"[Step 1] Loading Sleep-EDF subject {subject_idx}...")
    os.makedirs(data_dir, exist_ok=True)

    try:
        paths = mne.datasets.sleep_physionet.age.fetch_data(
            subjects=[subject_idx],
            recording=[1],
            path=data_dir,
            verbose=False,
        )
        psg_path, ann_path = paths[0]
        print(f" Loaded: {os.path.basename(psg_path)}")

        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
        raw.resample(FS, verbose=False)

        ch_names_upper = [c.upper() for c in raw.ch_names]
        eeg_chs = [raw.ch_names[i] for i, c in enumerate(ch_names_upper) if "EEG" in c]
        eog_chs = [raw.ch_names[i] for i, c in enumerate(ch_names_upper) if "EOG" in c]
        emg_chs = [raw.ch_names[i] for i, c in enumerate(ch_names_upper) if "EMG" in c or "CHIN" in c]
        eeg_ch = eeg_chs[0] if eeg_chs else raw.ch_names[0]
        eog_ch = eog_chs[0] if eog_chs else None
        emg_ch = emg_chs[0] if emg_chs else None
        print(f"  Using EEG={eeg_ch}, EOG={eog_ch}, EMG={emg_ch}")

        annotations = mne.read_annotations(ann_path, verbose=False)
        raw.set_annotations(annotations, verbose=False)

        events, event_id = mne.events_from_annotations(
            raw, event_id=STAGE_MAP, verbose=False
        )

        n_samples = int(EPOCH_SEC * FS) 
        records = []

        for i, (onset_sample, _, stage_code) in enumerate(events):
            if stage_code == -1:
                continue
            
            end_sample = onset_sample + n_samples
            
            if end_sample > len(raw.times):
                break

            eeg_data = raw.get_data(picks=eeg_ch, start=onset_sample, stop=end_sample)[0]
            eog_data = (raw.get_data(picks=eog_ch, start=onset_sample, stop=end_sample)[0] if eog_ch else None)
            emg_data = (raw.get_data(picks=emg_ch, start=onset_sample, stop=end_sample)[0] if emg_ch else None)

            feats = compute_features(eeg_data, eog_data, emg_data, FS)
            
            feats["epoch_id"] = i
            feats["true_label"] = stage_code
            feats["stage_name"] = STAGE_NAMES.get(stage_code, "?")
            records.append(feats)

        df = pd.DataFrame(records)
        print(f" Extracted {len(df)} epochs")
        print(f" Stage counts:\n{df['stage_name'].value_counts().to_string()}")

    except Exception as e:
        print(f" Download failed ({e}), using synthetic data...")
        df = _generate_synthetic_features(n_epochs=200)

    if save_csv:
        csv_path = os.path.join(data_dir, "features.csv")
        df.to_csv(csv_path, index=False)
        print(f" Saved to {csv_path}")

    return df

def _generate_synthetic_features(n_epochs=200):
    print("[FeatureExtractor] Generating synthetic data (offline mode)...")
    np.random.seed(42) 
    stage_profiles = {
        0: dict(delta=0.5, theta=0.3, alpha=2.0, beta=1.5, spindle=0.05, eog=0.8, emg=1.5, ent=3.5),
        1: dict(delta=1.0, theta=2.0, alpha=0.8, beta=0.3, spindle=0.08, eog=0.3, emg=0.5, ent=3.2),
        2: dict(delta=2.0, theta=1.0, alpha=0.3, beta=0.2, spindle=0.25, eog=0.1, emg=0.2, ent=2.8),
        3: dict(delta=8.0, theta=0.5, alpha=0.1, beta=0.05, spindle=0.05, eog=0.05,emg=0.1, ent=2.2),
        4: dict(delta=0.8, theta=1.8, alpha=0.4, beta=0.3, spindle=0.06, eog=1.5, emg=0.05,ent=3.3),
    }

    stage_weights = [0.15, 0.05, 0.45, 0.15, 0.20]
    stages = np.random.choice(5, size=n_epochs, p=stage_weights)

    records = []
    for i, s in enumerate(stages):
        p = stage_profiles[s]
        noise = 0.15
        row = {
            "epoch_id": i,
            "delta_power": max(0, np.random.normal(p["delta"], p["delta"]*noise)),
            "theta_power": max(0, np.random.normal(p["theta"], p["theta"]*noise)),
            "alpha_power": max(0, np.random.normal(p["alpha"], p["alpha"]*noise)),
            "beta_power": max(0, np.random.normal(p["beta"], p["beta"]*noise)),
            "spindle_ratio": max(0, np.random.normal(p["spindle"],0.03)),
            "eog_variance": max(0, np.random.normal(p["eog"], p["eog"]*noise)),
            "emg_variance": max(0, np.random.normal(p["emg"], p["emg"]*noise)),
            "signal_entropy": max(0, np.random.normal(p["ent"], 0.2)),
            "true_label": s,
            "stage_name": STAGE_NAMES[s],
        }
        records.append(row)

    df = pd.DataFrame(records)
    print(f" Generated {len(df)} epochs")
    print(f" Stage counts:\n{df['stage_name'].value_counts().to_string()}")
    return df

if __name__ == "__main__":
    df = load_and_extract(data_dir="data")
    print("\nFirst 5 rows:")
    print(df.head())