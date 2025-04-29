#! /usr/bin/env python3

from FillerDetector import FillerDetector
from audio_utils import audiostream
import argparse
import asyncio
import torch

async def classify_audiostream(classifier, torch_device, threshold=0.5, **kwargs):
    '''Asynchronous task to sample & classify audiostream data as it arrives'''
    print('Waiting for filler words...')
    filler_count = 0
    async for audio, _ in audiostream(**kwargs):
        data = torch.unsqueeze(torch.flatten(torch.from_numpy(audio)), 0)
        outputs = classifier(data.to(torch_device))
        probs = torch.sigmoid(outputs).view(-1).detach().numpy()
        is_filler = float(probs) > threshold
        filler_count += is_filler
        print(f'Filler detected! (total: {filler_count})') if is_filler else None

if __name__ == "__main__":
    # Parse arguments from command line
    parser = argparse.ArgumentParser(
        prog='bertha',
        description='Binary Estimator for Robust Temporal Hesitation Analysis'
    )
    parser.add_argument('model', help='path to model weights')
    parser.add_argument('-t', '--threshold', default=0.5, help='classifier decision threshold (1=filler)', type=float)
    parser.add_argument('-r', '--runtime', default=60, help='number of seconds the program should run', type=int)
    args = parser.parse_args()

    # Load the classifier model
    torch_device = torch.device('cpu')
    classifier = FillerDetector(out_dim=1)
    classifier.load_state_dict(torch.load(args.model, weights_only=True, map_location=torch_device))
    classifier.to(torch_device)

    async def classify_loop(timeout):
        try:
            await asyncio.wait_for(classify_audiostream(classifier, torch_device, args.threshold, blocksize=16000, samplerate=16000), timeout=timeout)
        except asyncio.TimeoutError:
            print('Timeout reached. Thank you for using BERTHA!')

    try:
        asyncio.run(classify_loop(args.runtime))
    except KeyboardInterrupt:
        print('Keyboard interrupt detected. Thank you for using BERTHA!')
