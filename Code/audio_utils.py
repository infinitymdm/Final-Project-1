#! /usr/bin/env python3

import asyncio
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
    '''Convert input audio (arbitrary length) to tensors of the desired shape. Returns a list of tensors'''
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
        print('min:', data.min(), '\tmax:', data.max())

if __name__ == "__main__":
    async def main():
        try:
            await asyncio.wait_for(sample_audiostream(), timeout=2)
        except asyncio.TimeoutError:
            pass

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')
