import asyncio
from elevenlabs import play
from pydub import AudioSegment
import pyaudio
import os
import websockets
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
SPEECH_ANALYZER_PORT = os.getenv("SPEECH_ANALYZER_PORT")


def save_audio_data(data: bytes, save_path: str = "output.wav") -> None:
    audio_segment = AudioSegment(
        data=data, sample_width=2, frame_rate=44100, channels=1
    )
    audio_segment.export(save_path, format="wav")
    logger.info(f"Saved file to {save_path}")


def record(duration: int = 3) -> bytes:
    py_audio = pyaudio.PyAudio()
    audio_stream = py_audio.open(
        rate=44100,  # frames per second,
        channels=1,  # stereo, change to 1 if you want mono
        format=pyaudio.paInt16,
        input=True,  # input device flag
        frames_per_buffer=1024,  # 1024 samples per frame
    )
    logger.info("Recording")
    frames = []
    for _ in range(int(44100 / 1024 * duration)):
        data = audio_stream.read(1024)
        frames.append(data)
    audio_stream.close()
    logger.info(py_audio.get_sample_size(pyaudio.paInt16))
    logger.info("Recording stop")
    return b"".join(frames)


async def send_file(filename: str) -> bytes:
    async with websockets.connect(
        f"ws://0.0.0.0:{SPEECH_ANALYZER_PORT}/ws"
    ) as websocket:
        with open(filename, "rb") as file:
            await websocket.send(file.read())
        while True:
            try:
                data = await websocket.recv()
                logger.info(f"Received data: len {len(data)} bytes")
                break
            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection closed")
                break  # Handle the case where the server closes the connection
    return data


async def send_microphone_data(audio_data: bytes):
    async with websockets.connect(
        f"ws://0.0.0.0:{SPEECH_ANALYZER_PORT}/ws"
    ) as websocket:
        logger.info(f"Sending data: {len(audio_data)}")
        await websocket.send(audio_data)
        while True:
            try:
                data = await websocket.recv()
                logger.info(f"Received data: len {len(data)} bytes")
                break
            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection closed")
                break  # Handle the case where the server closes the connection
    return data


if __name__ == "__main__":
    audio_bytes = record(duration=4)
    response = asyncio.run(send_microphone_data(audio_bytes))
    play(response)
