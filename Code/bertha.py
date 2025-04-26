#! /usr/bin/env python3

from FillerDetector import FillerDetector
from audio_utils import audiostream
import asyncio
import sys
import torch

async def classify_audiostream(classifier, torch_device, **kwargs):
    '''Asynchronous task to sample & classify audiostream data as it arrives'''
    async for audio, _ in audiostream(**kwargs):
        data = torch.unsqueeze(torch.flatten(torch.from_numpy(audio)), 0)
        outputs = classifier(data.to(torch_device))
        is_filler = outputs[0][1] > outputs[0][0]
        print(f'is_filler: {is_filler}')

if __name__ == "__main__":
    classifier = FillerDetector()
    classifier.load_state_dict(torch.load('ckpt/f0.840.ckpt/model.pt', weights_only=True))
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier.to(torch_device)

    async def classify_loop(timeout):
        try:
            await asyncio.wait_for(classify_audiostream(classifier, torch_device, blocksize=16000, samplerate=16000), timeout=timeout)
        except asyncio.TimeoutError:
            pass

    timeout = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    try:
        asyncio.run(classify_loop(timeout))
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')
