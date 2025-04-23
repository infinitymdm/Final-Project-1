#! /usr/bin/env python3

from FillerDetector import FillerDetector
from PodcastFillers import PodcastFillersDataset
from ignite.metrics import Fbeta
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train(model, train_data, loss_fn, optimizer, **hyperparams):
    '''Return a model trained on the dataset using the specified hyperparameters.

    Arguments:
        model:      a torch.nn.Module object to be trained
        dataset:    a torch.utils.data.DataLoader to be sampled for training data and labels
        loss_fn:    the loss function (e.g. `torch.nn.CrossEntropyLoss) to use in training
        optimizer:  the optimizer (e.g. `torch.optim.Adam`) to use in training

    Keyword arguments
        num_epochs: (defaults to 1)
        device:     (defaults to cuda if available)
    '''
    num_epochs = hyperparams.get('num_epochs', 1)
    device = hyperparams.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device)

    # Each epoch, iterate over data
    for epoch in range(num_epochs):
        model.train()
        avg_loss = 0
        with tqdm(total=len(train_data), desc=f'Epoch {epoch}') as progress_bar:
            for i, batch in enumerate(train_data):
                data, targets = batch

                # Perform one training step & calculate the gradient
                optimizer.zero_grad()
                predictions = model(data.to(device))
                loss = loss_fn(predictions, targets.to(device))
                loss.backward()
                optimizer.step()

                # Show progress
                avg_loss += loss
                progress_bar.update(1)
                progress_bar.set_postfix_str(f'Test Loss: {(avg_loss/i):.5f}')

    return model, optimizer


def test_fbeta(model, dataset, **hyperparams):
    '''Test a model on the given dataset and return the fbeta score.

    Arguments:
        model:      a torch.nn.Module to be tested
        dataset:    a torch.utils.data.DataLoader to be sampled for test data and labels

    Keyword arguments:
        beta:       (defaults to 1)
        device:     (defaults to cuda if available)
    '''
    device = hyperparams.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    beta = hyperparams.get('beta', 1)

    # Initialize the fbeta metric to average over the test dataset
    fbeta = Fbeta(beta=beta, average=True)

    # Iterate over data
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_data), desc='Testing') as progress_bar:
            for i, batch in enumerate(test_data):
                data, targets = batch

                # Test on the batch of data and calculate fbeta score
                predictions = model(data.to(device))
                fbeta.update((predictions, targets.to(device)))
                score = fbeta.compute()

                # Show progress
                progress_bar.update(1)
                progress_bar.set_postfix_str(f'f{beta}: {score:.5f}')

    return score


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 16
    learn_rate = 1e-3
    num_epochs = 1
    loss_fn = torch.nn.CrossEntropyLoss()
    opt_fn = torch.optim.Adam
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize train and test datasets
    pcf_root = Path('data/PodcastFillers')
    pcf_csv = pcf_root / 'metadata' / 'PodcastFillers.csv'
    pcf_wav_dir = pcf_root / 'audio' / 'clip_wav'
    pcf_dataset = lambda s: PodcastFillersDataset(pcf_csv, pcf_wav_dir, split=s)
    train_data = DataLoader(pcf_dataset('train'), batch_size=batch_size, shuffle=True)
    test_data = DataLoader(pcf_dataset('test'), batch_size=batch_size)

    # Initialize the model and optimizer
    model = FillerDetector()
    model.to(device)
    optimizer = opt_fn(model.parameters(), lr=learn_rate)

    # If loading a previous checkpoint, set ckpt_name to the filepath
    ckpt_dir = Path('ckpt')
    ckpt_name = ckpt_dir / 'f0.838.ckpt'
    if ckpt_name.exists():
        print(f'Loading model {ckpt_name} for continued training...')
        model.load_state_dict(torch.load(ckpt_name / 'model.pt', weights_only=True))
        model.to(device)
        optimizer.load_state_dict(torch.load(ckpt_name / 'optimizer.pt'))

    # Train the model & evaluate results
    model, optimizer = train(model, train_data, loss_fn, optimizer, num_epochs=num_epochs)
    score = test_fbeta(model, test_data, beta=0.5)

    # Save a checkpoint
    ckpt_name = ckpt_dir / f'f{score:.3f}.ckpt'
    ckpt_name.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_name / 'model.pt')
    torch.save(optimizer.state_dict(), ckpt_name / 'optimizer.pt')
