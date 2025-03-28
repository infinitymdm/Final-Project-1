# ECEN5060 Final Project

This repository contains code and documents for a deep network which identifies filler words in
real-time speech audio. This network is designed as a course project in Dr. Hagan's Deep Learning
course.

## Overview

- Given live speech input, the network should indicate if the last few seconds (ish?) of audio contained a filler word
- Trained on the [TED-LIUM dataset](https://huggingface.co/datasets/LIUM/tedlium)
- We'll need to select a network architecture that has memory (recurrent, LSTM, autoencoder, transformer, etc.)
- We'll need to decide on performance metrics, but at the end of the day we probably want a higher FN rate than FP

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
pip install datasets librosa sounddevice soundfile
```

## Resources

- [Typst Documentation](https://typst.app/docs)
- [TED-LIUM dataset](https://huggingface.co/datasets/LIUM/tedlium)
