#! /usr/bin/env python3

from FillerDetector import FillerDetector
from PodcastFillers import PodcastFillersDataset
from ignite.metrics import Fbeta
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

    # Initialize a loader for the dataset
    train_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the optimizer with the model parameters (on the appropriate device)
    model.to(device)
    optimizer = opt_fn(model.parameters(), lr=learn_rate)

    # Each epoch, iterate over data
    for epoch in range(num_epochs):
        model.train()
        avg_loss = 0
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
                avg_loss += loss
                progress_bar.update(1)
                progress_bar.set_postfix_str(f'Test Loss: {(avg_loss/i):.5f}')

    return model


def test_fbeta(model, dataset, **hyperparams):
    '''Test a model on the given dataset and return the fbeta score.

    Arguments:
        model:      a torch.nn.Module to be tested
        dataset:    a torch.utils.data.Dataset to be sampled for test data and labels

    Keyword arguments:
        batch_size: (defaults to 8)
        device:     (defaults to cuda if available)
        beta:       (defaults to 1)
    '''
    batch_size = hyperparams.get('batch_size', 8)
    device = hyperparams.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    beta = hyperparams.get('beta', 1)

    # Initialize a loader for the dataset
    test_data = DataLoader(dataset, batch_size=batch_size)

    # Initialize the fbeta metric to average over the test dataset
    fbeta = Fbeta(beta=beta, average=True)

    # Iterate over data
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_data), desc='Testing') as progress_bar:
            for i, batch in enumerate(test_data):
                data, targets = batch
                data.to(device)
                targets.to(device)

                # Test on the batch of data and calculate fbeta score
                predictions = model(data)
                fbeta.update((predictions, targets))
                score = fbeta.compute()

                # Show progress
                progress_bar.update(1)
                progress_bar.set_postfix_str(f'f{beta}: {score:.5f}')

    return score


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32

    # Initialize train and test datasets
    pcf_root = Path('data/PodcastFillers')
    pcf_csv = pcf_root / 'metadata' / 'PodcastFillers.csv'
    pcf_wav_dir = pcf_root / 'audio' / 'clip_wav'
    pcf_dataset = lambda s: PodcastFillersDataset(pcf_csv, pcf_wav_dir, split=s)

    # Initialize the model
    model = torch.compile(FillerDetector())

    # Train the model
    model = train(model, pcf_dataset('train'), torch.nn.CrossEntropyLoss(), torch.optim.Adam, batch_size=batch_size)

    # Test the model
    score = test_fbeta(model, pcf_dataset('test'), beta=0.5, batch_size=batch_size)

    # Save the model to a file
    torch.save(model.state_dict(), f'model_{score}.pt')
