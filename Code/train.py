#! /usr/bin/env python3

from FillerDetector import FillerDetector
from PodcastFillers import PodcastFillersDataset
from ignite.metrics import Fbeta
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd

def train(model, train_data, loss_fn, optimizer, intermediate_test=None, **hyperparams):
    '''Return a model trained on the dataset using the specified hyperparameters.

    Arguments:
        model:      a torch.nn.Module object to be trained
        dataset:    a torch.utils.data.DataLoader to be sampled for training data and labels
        loss_fn:    the loss function (e.g. `torch.nn.CrossEntropyLoss) to use in training
        optimizer:  the optimizer (e.g. `torch.optim.Adam`) to use in training
        intermediate_test: an optional function to save the intermediate model with the best accuracy

    Keyword arguments
        num_epochs: (defaults to 1)
        device:     (defaults to cuda if available)
        beta:       (defaults to 1)
    '''
    num_epochs = hyperparams.get('num_epochs', 1)
    device = hyperparams.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device)

    ckpt_dir = Path('ckpt')
    best_score = 0.0
    beta = hyperparams.get('beta', 0.5)

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

        # Save the best performing (f-0.5 score) model at intermediate steps
        if intermediate_test is not None:
            score, _ = test_fbeta(model, intermediate_test, beta=beta, device=device)
            if score > best_score:
                best_score = score
                ckpt_name = ckpt_dir / f'f{score:.3f}_epoch{epoch}.ckpt'
                ckpt_name.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_name / 'model.pt')
                torch.save(optimizer.state_dict(), ckpt_name / 'optimizer.pt')
                print(f"New best score: {best_score:.3f} saved to {ckpt_name}")

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

    all_predictions = []
    all_targets = []

    # Iterate over data
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_data), desc='Testing') as progress_bar:
            for i, batch in enumerate(test_data):
                data, targets = batch

                # Test on the batch of data and calculate fbeta score
                logits = model(data.to(device))
                pred_batch = torch.argmax(logits, dim=1).cpu().numpy()
                targ_batch = targets.cpu().numpy()
                fbeta.update((logits, targets.to(device)))
                score = fbeta.compute()

                all_predictions.extend(pred_batch)
                all_targets.extend(targ_batch)

                # Show progress
                progress_bar.update(1)
                progress_bar.set_postfix_str(f'f{beta}: {score:.5f}')

    cm = confusion_matrix(all_targets, all_predictions)
    return score, cm


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 16
    learn_rate = 1e-3
    num_epochs = 50
    opt_fn = torch.optim.Adam
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # The custom loss weights punish false positives twice as much as false negatives so our detector is not likely to
    # constantly "ring the bell" by detecting a filler.
    custom_loss_weights = torch.tensor([3.5, 1.0],device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=custom_loss_weights)

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
    ckpt_name = ckpt_dir / '0.ckt'#'f0.864_epoch47.ckpt'
    if ckpt_name.exists():
        print(f'Loading model {ckpt_name} for continued training...')
        model.load_state_dict(torch.load(ckpt_name / 'model.pt', weights_only=True))
        model.to(device)
        optimizer.load_state_dict(torch.load(ckpt_name / 'optimizer.pt'))

    # Train the model & evaluate results
    model, optimizer = train(model, train_data, loss_fn, optimizer, intermediate_test=test_data, num_epochs=num_epochs)
    score, cm = test_fbeta(model, test_data, beta=0.5)
    #print(model)
    print(f"F0.5 Score: {score:.5f}")
    print(pd.DataFrame(
        cm,
        index=[f"True Speech", f"True Filler"],
        columns=[f"Pred Speech", f"Pred Filler"]
    ))

    # Save a checkpoint
    # AA: Commented this section out as I now save model on the fly to save the best performance rather than last model
    '''ckpt_name = ckpt_dir / f'f{score:.3f}.ckpt'
    ckpt_name.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_name / 'model.pt')
    torch.save(optimizer.state_dict(), ckpt_name / 'optimizer.pt')'''
