#! /usr/bin/env python3

from FillerDetector import FillerDetector
import asyncio
import sounddevice as sd
# See https://python-sounddevice.readthedocs.io/en/0.5.1/usage.html#callback-streams for info
# on real-time recording & playback
import torch

def record(audio_device, duration):
    '''Record audio from audio_device for duration. Returns audio, sample_rate'''
    pass

def read(audio_file):
    '''Read audio from a file. Returns audio, sample_rate'''
    pass

def convert_to_tensors(audio, sample_rate, shape):
    '''Convert input audio of arbitrary length to tensors of the desired shape. Returns a list of tensors'''
    pass

async def audiostream(channels=1, **kwargs):
    '''Generator yielding blocks of input audio data'''
    q_in = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def callback(data, frame_count, time_info, status):
        loop.call_soon_threadsafe(q_in.put_nowait, (data.copy(), status))

    stream = sd.InputStream(callback=callback, channels=channels, **kwargs)
    with stream:
        while True:
            data, status = await q_in.get()
            yield data, status

async def sample_audiostream(**kwargs):
    '''Asynchronous task to sample data from an audiostream'''
    async for data, status in audiostream(**kwargs):
        if status:
            print(status)
        print(data.shape)

async def classify_audiostream(classifier, torch_device, **kwargs):
    '''Asynchronous task to sample & classify audiostream data as it arrives'''
    async for audio, _ in audiostream(**kwargs):
        data = torch.unsqueeze(torch.flatten(torch.from_numpy(audio)), 0)
        outputs = classifier(data.to(torch_device))
        print(outputs)

if __name__ == "__main__":
    classifier = FillerDetector()
    classifier.load_state_dict(torch.load('ckpt/f0.838.ckpt/model.pt', weights_only=True))
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier.to(torch_device)

    async def main():
        try:
            await asyncio.wait_for(classify_audiostream(classifier, torch_device, blocksize=16000, samplerate=16000), timeout=10)
        except asyncio.TimeoutError:
            pass

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')
