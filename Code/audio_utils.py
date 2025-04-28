#! /usr/bin/env python3

import asyncio
import sounddevice as sd
# See https://python-sounddevice.readthedocs.io/en/0.5.1/usage.html#callback-streams for info
# on real-time recording & playback
import torch

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
