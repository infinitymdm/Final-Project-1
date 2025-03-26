#! /usr/bin/env python3

import librosa
import time
import numpy as np
import sounddevice as sd
from datasets import load_dataset

tedlium = load_dataset("LIUM/tedlium", "release3")


for i in range(2863, 3000):
    sample = tedlium['train'][i]
    if 'fill' in sample['text'].lower():
        print(i)
        break

print(sample['text'])
audio = sample['audio']['array']
sr = sample['audio']['sampling_rate']
dur = librosa.get_duration(y=audio, sr=sr)
sd.play(audio, sr)
time.sleep(dur)
# [print(tedlium["train"][i]["text"]) for i in range(50, 70)]
