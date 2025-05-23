#! /usr/bin/env python3

from AA_FillerDetector import FillerDetector
from AA_PodcastFillers import PodcastFillersDataset
from ignite.metrics import Fbeta
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import pandas as pd
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

# This method is a big mix of Marcus and Andrew code very intermingled, the deeper into the method you go the more it is
# Andrew code from the various modifications required thorughout the design process
def train(model, train_data, loss_fn, optimizer, scheduler=None, intermediate_test=None, run_dir=Path(''), **hyperparams):
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
        patience:     (defaults to 5)
    '''
    num_epochs = hyperparams.get('num_epochs', 1)
    device = hyperparams.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device)

    ckpt_dir = Path('ckpt')
    best_score = 0.0
    beta = hyperparams.get('beta', 0.5)
    epoch_metrics = []

    # Andrew added variables
    # Early stopping variables. If no improvements occur in 5 epochs, the training stops early to avoid wasted training.
    no_change = 0
    patience = hyperparams.get('patience', 8)


    # Each epoch, iterate over data
    for epoch in range(num_epochs):
        model.train()
        avg_loss = 0
        with tqdm(total=len(train_data), desc=f'Epoch {epoch}') as progress_bar:
            for i, batch in enumerate(train_data):
                data, targets = batch

                targets = targets.float().to(device).view(-1)

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

        # From this point on it is entirely Andrew code
        # Save the best performing (f-0.5 score) model at intermediate steps
        if intermediate_test is not None:
            f05, _, acc, prec, rec = test_metrics(model, intermediate_test, beta=beta, device=device)

            if scheduler is not None:
                scheduler.step(f05)

            ckpt_name = run_dir / f'f{f05:.3f}_epoch{epoch}'
            ckpt_name.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_name / 'model.pt')
            torch.save(optimizer.state_dict(), ckpt_name / 'optimizer.pt')

            if f05 > best_score:
                best_score = f05
                no_change = 0
                print(f"New best score: {best_score:.3f} saved to {ckpt_name}")
            else:
                no_change += 1

            epoch_metrics.append({
                "epoch": epoch,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f0.5": f05
            })

            if no_change >= patience:
                print(f"Early stopping at epoch {epoch} — no improvement in {patience} epochs.")
                break

    df_metrics = pd.DataFrame(epoch_metrics)
    df_metrics.to_csv(run_dir / "metrics.csv", index=False)

    return model, optimizer

# AA: This is a legacy method for tracking changes in the script. I do not recommend using as it can make large negative losses that destabilize training.
def CustomBCE(logits, targets, fp_weight=1.0,fn_weight=1.0):
    probs = torch.sigmoid(logits)
    eps = 1e-6
    loss = -((fn_weight * targets * torch.log(probs + eps)) + (fp_weight * (1 - targets) * torch.log(1 - probs + eps)))
    return loss.mean()

# This is an Andrew written method built on top of what was originally test_Fbeta (Marcus code).
# Any lines that look different, are Andrew written, if they are the same between the two, those are Marcus written.
def test_metrics(model, test_data, **hyperparams):
    '''Test a model on the given dataset and return the fbeta score, precision, recall, and accuracy.

        Arguments:
            model:      a torch.nn.Module to be tested
            test_data:    a torch.utils.data.DataLoader to be sampled for test data and labels

        Keyword arguments:
            beta:       (defaults to 1)
            device:     (defaults to cuda if available)
            threshold:  (defaults to 0.5)
        '''
    device = hyperparams.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    beta = hyperparams.get('beta', 0.5)
    threshold = hyperparams.get('threshold', 0.5)

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
                # probs in this function is performing a sigmoid under the assumption of a single neuron output layer
                probs = torch.sigmoid(logits).view(-1).cpu().numpy()
                pred_batch = (probs > threshold).astype(int)  # Custom threshold if needed

                targ_batch = targets.cpu().numpy()
                fbeta.update((torch.tensor(pred_batch).to(device), targets.to(device)))
                score = fbeta.compute()

                all_predictions.extend(pred_batch)
                all_targets.extend(targ_batch)

                # Show progress
                progress_bar.update(1)
                progress_bar.set_postfix_str(f'f{beta}: {score:.5f}')

    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)

    cm = confusion_matrix(all_targets, all_predictions)
    return score, cm, accuracy, precision, recall

# A Marcus legacy method, no longer used in the train script, but kept for continuity
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

# This is another block with occasional lines written by Marcus particularly at the top with quite a lot of modification
# made by Andrew throughout the project.
if __name__ == "__main__":
    now = datetime.now()
    run_name = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = Path("ckpt") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving run to: {run_dir}")

    # Hyperparameters
    batch_size = 16
    learn_rate = 1e-3 # For CEL this rate works well
    num_epochs = 50
    opt_fn = torch.optim.Adam
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # The custom loss weights punish false positives twice as much as false negatives so our detector is not likely to
    # constantly "ring the bell" by detecting a filler.
    # A value of 1.5 in the weight does slightly improve the false positive rate, but over many epochs can actually start to do the opposite
    # A value of 3.5 in the weight greatly reduces the false positive rate at the cost of doubling the false negative rate
    # Values in between are some combination of the two tradeoffs. The best performance so far has led to 1/8 positives are false and 1/7 negatives are false
    '''
    # loss weights for a CEL setup
    custom_loss_weights = torch.tensor([3.5, 1.0],device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=custom_loss_weights)
    '''
    pos_weight = torch.tensor([0.75], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Initialize train and test datasets
    pcf_root = Path('data/PodcastFillers')
    pcf_csv = pcf_root / 'metadata' / 'PodcastFillers.csv'
    pcf_wav_dir = pcf_root / 'audio' / 'clip_wav'
    pcf_dataset = lambda s: PodcastFillersDataset(pcf_csv, pcf_wav_dir, split=s, max_shift=2400)
    train_data = DataLoader(pcf_dataset('train'), batch_size=batch_size, shuffle=True)
    test_data = DataLoader(pcf_dataset('test'), batch_size=batch_size)
    val_data = DataLoader(pcf_dataset('validation'), batch_size=batch_size, shuffle=True)

    # Initialize the model and optimizer
    model = FillerDetector(out_dim=1)
    model.to(device)
    optimizer = opt_fn(model.parameters(), lr=learn_rate)

    # Include a learning rate scheduler to help training be more focused on the f0.5 score
    # (or other metric the gradient cannot directly train to)
    # by reducing learning rate if we don't improve target metric often
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.33)

    # If loading a previous checkpoint, set ckpt_name to the filepath
    ckpt_dir = Path('ckpt')
    ckpt_name = ckpt_dir / '0.ckpt'
    if ckpt_name.exists():
        print(f'Loading model {ckpt_name} for continued training...')
        model.load_state_dict(torch.load(ckpt_name / 'model.pt', weights_only=True))
        model.to(device)
        optimizer.load_state_dict(torch.load(ckpt_name / 'optimizer.pt'))

    # Print the model structure
    #print(model)

    # Train the model & evaluate results
    model, optimizer = train(model, train_data, loss_fn, optimizer, scheduler=lr_scheduler, intermediate_test=val_data, run_dir=run_dir, num_epochs=num_epochs, patience=10)
    score, cm, _, _, _ = test_metrics(model, test_data, beta=0.5)
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
