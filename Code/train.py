#! /usr/bin/env python3

from FillerDetector import FillerDetector
from PodcastFillers import PodcastFillersDataset
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import tqdm

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 8
    learn_rate = 1e-3
    num_epochs = 3
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=learn_rate)

    # Set up pytorch to use hardware acceleration (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize train and test datasets
    pcf_root = Path('data/PodcastFillers')
    pcf_csv = pcf_root / 'metadata' / 'PodcastFillers.csv'
    pcf_wav_dir = pcf_root / 'audio' / 'clip_wav'
    pcf_dataset = lambda s: PodcastFillersDataset(pcf_csv, pcf_wav_dir, split=s)
    train_data = DataLoader(pcf_dataset('train'), batch_size=batch_size, shuffle=True)
    test_data = DataLoader(pcf_dataset('test'), batch_size=batch_size)

    # Initialize the model
    model = FillerDetector()

    # Train the model
    for epoch in range(num_epochs):
        with tqdm(total=len(train_data), desc=f'Epoch {epoch}') as progress_bar:
            for inputs, labels in train_data:
                # Perform one training step & calculate the gradient
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                progress_bar.update(1)
                progress_bar.set_postfix_str(f'Test Loss: {loss:.5f}')
