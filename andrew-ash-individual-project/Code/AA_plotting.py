import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# This boilerplate plotting code was written by ChatGPT,
# I (Andrew) then edited the formatting to achieve desired formatting and directory structure for saving

# ------------------ CONFIG ------------------
# Set the run directory where "metrics.csv" was saved
run_dir = Path("ckpt/2025-04-27_22-00")
metrics_csv = run_dir / "metrics.csv"

# ------------------ LOAD DATA ------------------
df = pd.read_csv(metrics_csv)

# ------------------ PLOTTING ------------------

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# F0.5 score
axs[0, 0].plot(df['epoch'], df['f0.5'], marker='o')
axs[0, 0].set_title('F0.5 Score over Epochs')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('F0.5 Score')
axs[0, 0].grid()

# Accuracy
axs[0, 1].plot(df['epoch'], df['accuracy'], marker='o')
axs[0, 1].set_title('Accuracy over Epochs')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Accuracy')
axs[0, 1].grid()

# Precision
axs[1, 0].plot(df['epoch'], df['precision'], marker='o')
axs[1, 0].set_title('Precision over Epochs')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Precision')
axs[1, 0].grid()

# Recall
axs[1, 1].plot(df['epoch'], df['recall'], marker='o')
axs[1, 1].set_title('Recall over Epochs')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('Recall')
axs[1, 1].grid()

plt.tight_layout()
plt.suptitle(f"Training Metrics for {run_dir.name}", y=1.02)

# ------------------ SAVE THE PLOT ------------------
plot_file = run_dir / "metrics_plot.png"
plt.savefig(plot_file, bbox_inches='tight')
print(f"Plot saved to: {plot_file}")

# ------------------ SHOW THE PLOT ------------------
plt.show()