# ECEN5060 Final Project

This repository contains code and documents for a deep network which identifies filler words in
real-time speech audio. This network is designed as a course project in Dr. Hagan's Deep Learning
course.

## Overview

- Given live speech input, detects filler words and provides real-time feedback with upwards of 90% accuracy
- Trained on the [PodcastFillers](https://podcastfillers.github.io/) dataset
- Multiple trained models with adjustable detection thresholds

## Organization

- `Proposal` contains typst source for the project proposal
- `Final-Group-Project-Report` contains typst source for the team report
- `Final-Presentation` contains presentation slides
- `Code` contains code used to implement and train the network

## Dependencies

To run the code in this repository, you'll need a few dependencies. We recommend setting up an
isolated python environment with

```sh
python3 -m venv .venv # Create a virtual python environment called .venv
source .venv/bin/activate # Activate the virtual environment
```

Then you can install the dependencies:

```sh
pip install matplotlib numpy pandas pytorch-ignite scikit-learn sounddevice torchaudio tqdm
```

## Running the Code

After cloning the repository, run `bertha.py -h` from the Code directory to view usage information.
