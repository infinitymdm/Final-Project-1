#! /usr/bin/env python3

import sounddevice as sd
# See https://python-sounddevice.readthedocs.io/en/0.5.1/usage.html#callback-streams for info
# on real-time recording & playback
# Maybe we can use a Stream callback to produce 1-second chunks of audio?
# See also https://python-sounddevice.readthedocs.io/en/0.5.1/examples.html#plot-microphone-signal-s-in-real-time

def record(audio_device, duration):
    '''Record audio from audio_device for duration. Returns audio, sample_rate'''
    pass

def read(audio_file):
    '''Read audio from a file. Returns audio, sample_rate'''
    pass

def downsample(audio, sample_rate, new_sample_rate):
    '''Downsample audio to a new sample rate'''
    pass

def convert_to_tensors(audio, sample_rate, shape):
    '''Convert input audio to tensors of the desired shape. Returns a list of tensors'''
    pass
