#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas
from pathlib import Path
import torch
import torchaudio

# Important constants, might need these in other files
recognized_fillers = {'uh', 'um', 'you know', 'other', 'like'}
recognized_nonfillers = {'words', 'repetitions', 'breath', 'laughter', 'music', 'agree', 'noise', 'overlap'}

# Things to consider:
# - Maybe use spectrograms instead of raw audio?

def relabel_fillers_bool(annotation):
    '''
    Given a raw dataframe from the annotations file, return whether the label is a filler word
    '''
    # Use consolidated vocab? No
    label = str(annotation['label_full_vocab']).lower()
    return label in recognized_fillers


class PodcastFillersDataset(torch.utils.data.Dataset):
    '''
    Custom dataset for PodcastFillers labeled wav clips

    Based on https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    '''

    def __init__(self, annotations_csv: str, wav_dir: str, split: str, transform=None, target_transform=relabel_fillers_bool):
        '''
        Construct the dataset

        Args:
            annotations_csv (str or path-like): path to the annotations csv file
            wav_dir (str or path-like): path to the 1-second clipped wav files
            split (str): which split to use (extra, test, train, validation)
            transform (callable): a transformation to apply to audio data (defaults to none)
            target_transform (callable): a transformation to apply to target labels (defaults to relabeling "filler/nonfiller")
        '''
        unsplit_annotations = pandas.read_csv(annotations_csv)
        self.annotations = unsplit_annotations[unsplit_annotations['clip_split_subset'] == split]
        self.wav_dir = Path(wav_dir) / split
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        '''
        Return the number of annotated clips in the dataset

        This is a magic method for the "len(dataset)" syntax
        '''
        return len(self.annotations)

    def __getitem__(self, index):
        '''
        Return a single audio clip and the associated annotation by index

        This is a magic method for the "dataset[index]" syntax
        '''
        filename = str(self.annotations.iloc[index, 0]) # col 0 is 'clip_name'
        wav_file = self.wav_dir / filename
        audio, sample_rate = torchaudio.load(wav_file)
        annotation = self.annotations.iloc[index, 1:] # use col 1 through end as annotations
        if self.transform:
            audio, sample_rate = self.transform(audio, sample_rate)
        if self.target_transform:
            annotation = self.target_transform(annotation)
        return audio, annotation

def plot_audio(audio, sample_rate=16000):
    '''
    Plot the waveform and spectrogram for an audio sample

    Mostly copied from https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html#loading-audio-data
    '''
    waveform = audio.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    # Set up subplots with waveforms above, spectrograms below
    figure, axes = plt.subplots(2, num_channels, sharex=True)

    # Plot by channel
    for c in range(num_channels):
        # Plot waveform
        axes[c].plot(time_axis, waveform[c])
        axes[c].grid()
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')

        # Plot spectrogram
        axes[c+1].specgram(waveform[c], Fs=sample_rate)


# If run as entry point, demo the dataset
if __name__ == "__main__":
    # Construct the dataset and set up a dataloader
    pcf_root = Path('../data/PodcastFillers')
    train_data = PodcastFillersDataset(pcf_root / 'metadata/PodcastFillers.csv', pcf_root / 'audio/clip_wav', split='train')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)

    features, labels = next(iter(train_loader))
    for i in range(8):
        audio = features[i]
        label = labels[i]
        plot_audio(audio)
        print(f'Label: {label}')
        plt.show()
