import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, fbeta_score
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from FillerDetector import FillerDetector
from PodcastFillers import PodcastFillersDataset
from torch.utils.data import DataLoader

def sweep_thresholds(model, val_data, device, save_dir, beta=0.5):
    model.eval()
    model.to(device)

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for data, targets in tqdm(val_data, desc="Sweeping thresholds"):
            data = data.to(device)
            logits = model(data)
            probs = torch.sigmoid(logits).view(-1).cpu().numpy()
            targets = targets.view(-1).cpu().numpy()

            all_probs.extend(probs)
            all_targets.extend(targets)

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # Calculate precision, recall, thresholds
    precision, recall, thresholds = precision_recall_curve(all_targets, all_probs)

    # Calculate F-beta scores at each threshold
    f05_scores = []
    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        f05 = fbeta_score(all_targets, preds, beta=beta)
        f05_scores.append(f05)

    f05_scores = np.array(f05_scores)

    # Find best threshold
    best_idx = np.argmax(f05_scores)
    best_threshold = thresholds[best_idx]
    best_f05 = f05_scores[best_idx]

    print(f"Best Threshold: {best_threshold:.3f} with F{beta}: {best_f05:.4f}")

    # Save thresholds and scores to CSV
    results_df = pd.DataFrame({
        'threshold': thresholds,
        'precision': precision[:-1],  # precision/recall are 1 longer than thresholds
        'recall': recall[:-1],
        'f0.5': f05_scores
    })
    results_df.to_csv(Path(save_dir) / "threshold_sweep.csv", index=False)

    # Plot precision, recall, F0.5 vs threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.plot(thresholds, recall[:-1], label="Recall")
    plt.plot(thresholds, f05_scores, label=f"F{beta}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Precision / Recall / F{beta} vs Threshold")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(Path(save_dir) / "precision-recall-curve.png")
    plt.close()

    return best_threshold, best_f05


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pcf_root = Path('data/PodcastFillers')
    pcf_csv = pcf_root / 'metadata' / 'PodcastFillers.csv'
    pcf_wav_dir = pcf_root / 'audio' / 'clip_wav'
    pcf_dataset = lambda s: PodcastFillersDataset(pcf_csv, pcf_wav_dir, split=s, max_shift=2400)
    val_data = DataLoader(pcf_dataset('validation'), batch_size=batch_size, shuffle=True)

    # Initialize the model and optimizer
    model = FillerDetector(out_dim=1)
    model.to(device)

    # If loading a previous checkpoint, set ckpt_name to the filepath
    ckpt_dir = Path('ckpt')
    ckpt_dir = ckpt_dir / '2025-04-27_22-00'
    ckpt_name = ckpt_dir / 'f0.889_epoch16'
    if ckpt_name.exists():
        print(f'Loading model {ckpt_name} for continued training...')
        model.load_state_dict(torch.load(ckpt_name / 'model.pt', weights_only=True))
        model.to(device)

    best_threshold, best_f05 = sweep_thresholds(model, val_data, device, save_dir=ckpt_dir, beta=0.5)
