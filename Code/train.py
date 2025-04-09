#! /usr/bin/env python3

from FillerDetector import FillerDetector
from PodcastFillers import PodcastFillersDataset
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train(model, dataset, loss_fn, opt_fn, **hyperparams):
    '''Return a model trained on the dataset using the specified hyperparameters.

    Arguments:
        model:      a torch.nn.Module object to be trained
        dataset:    a torch.utils.data.Dataset to be sampled for training data and labels
        loss_fn:    the loss function (e.g. `torch.nn.CrossEntropyLoss()`) to use in training
        opt_fn:     the optimizer function (e.g. `torch.optim.Adam`) to use in training

    Keyword arguments
        batch_size: (defaults to 8)
        learn_rate: (defaults to 1e-3)
        num_epochs: (defaults to 1)
        device:     (defaults to cuda if available)
    '''
    batch_size = hyperparams.get('batch_size', 8)
    learn_rate = hyperparams.get('learn_rate', 1e-3)
    num_epochs = hyperparams.get('num_epochs', 1)
    device = hyperparams.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Initialize the optimizer with the model parameters (on the appropriate device)
    model.to(device)
    optimizer = opt_fn(model.parameters(), lr=learn_rate)

    # Main training loop
    for epoch in range(num_epochs):
        with tqdm(total=len(train_data), desc=f'Epoch {epoch}') as progress_bar:
            for i, batch in enumerate(train_data):
                data, targets = batch
                data.to(device)
                targets.to(device)

                # Perform one training step & calculate the gradient
                optimizer.zero_grad()
                predictions = model(data)
                loss = loss_fn(predictions, targets)
                loss.backward()
                optimizer.step()

                # Show progress
                progress_bar.update(1)
                progress_bar.set_postfix_str(f'Test Loss: {loss:.5f}')

    return model


if __name__ == "__main__":
    # Initialize train and test datasets
    pcf_root = Path('data/PodcastFillers')
    pcf_csv = pcf_root / 'metadata' / 'PodcastFillers.csv'
    pcf_wav_dir = pcf_root / 'audio' / 'clip_wav'
    pcf_dataset = lambda s: PodcastFillersDataset(pcf_csv, pcf_wav_dir, split=s)
    train_data = DataLoader(pcf_dataset('train'), batch_size=batch_size, shuffle=True)
    test_data = DataLoader(pcf_dataset('test'), batch_size=batch_size)

    # Initialize the model
    model = FillerDetector(train_transfer_model)

    # Train the model
    train(model, train_data, torch.nn.CrossEntropyLoss(), torch.optim.Adam, batch_size=32)

    # Save the model to a file
    torch.save(model.state_dict(), "model.pt")
