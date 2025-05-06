from AA_FillerDetector import FillerDetector
from AA_PodcastFillers import PodcastFillersDataset
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from ignite.metrics import Fbeta
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# AA: This boilerplate script was written by GPT to cut out all training from the training script for fast and easy testing of old models.
# I then modified the script as needed to meet all of my needs for quick testing.

# ------------------ CONFIG ------------------
# Set the path to the checkpoint directory
ckpt_name = Path('ckpt/2025-04-26_00-52/f0.854_epoch49')

# Model hyperparameters
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test dataset paths
pcf_root = Path('data/PodcastFillers')
pcf_csv = pcf_root / 'metadata' / 'PodcastFillers.csv'
pcf_wav_dir = pcf_root / 'audio' / 'clip_wav'

# Threshold for deciding positive/negative
threshold = 0.5

# ---------------------------------------------

def test_metrics(model, dataset, beta=0.5, threshold=0.5, device='cpu'):
    fbeta = Fbeta(beta=beta, average=True)

    all_predictions = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(dataset), desc='Testing') as progress_bar:
            for batch in dataset:
                data, targets = batch
                logits = model(data.to(device))
                probs = torch.sigmoid(logits).view(-1).cpu().numpy()
                pred_batch = (probs > threshold).astype(int)
                targ_batch = targets.cpu().numpy()

                fbeta.update((torch.tensor(pred_batch).to(device), targets.to(device)))

                all_predictions.extend(pred_batch)
                all_targets.extend(targ_batch)

                progress_bar.update(1)

    score = fbeta.compute()
    cm = confusion_matrix(all_targets, all_predictions)
    acc = accuracy_score(all_targets, all_predictions)
    prec = precision_score(all_targets, all_predictions, zero_division=0)
    rec = recall_score(all_targets, all_predictions, zero_division=0)
    return score, cm, acc, prec, rec

# ------------------ MAIN ------------------
if __name__ == "__main__":
    # Load dataset
    test_data = DataLoader(
        PodcastFillersDataset(pcf_csv, pcf_wav_dir, split='test'),
        batch_size=batch_size
    )

    # Load model
    model = FillerDetector(out_dim=1)
    model.load_state_dict(torch.load(ckpt_name / 'model.pt', weights_only=True))
    model.to(device)
    model.eval()

    # Evaluate
    score, cm, acc, prec, rec = test_metrics(model, test_data, beta=0.5, threshold=threshold, device=device)

    # Print results
    print(f"F0.5 Score: {score:.5f}")
    print(f"Accuracy: {acc:.5f}")
    print(f"Precision: {prec:.5f}")
    print(f"Recall: {rec:.5f}")
    print(pd.DataFrame(
        cm,
        index=["True Speech", "True Filler"],
        columns=["Pred Speech", "Pred Filler"]
    ))
