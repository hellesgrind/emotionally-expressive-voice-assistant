import os
import time
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from pydantic import BaseModel
from openai import AsyncOpenAI

from logs import logger

load_dotenv()

WHISPER_CLIENT = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Transcription(BaseModel):
    text: str


class WhisperSTT:
    def __init__(
        self,
        model: str,
        language: str,
    ):
        self.model = model
        self.language = language
        self.client = WHISPER_CLIENT

    async def transcribe_audio(self, file_path: str) -> str:
        audio_file = open(file_path, "rb")
        response = await self.client.audio.transcriptions.create(
            model=self.model,
            language=self.language,
            file=audio_file,
        )
        return response.text


class BaseTranscriptor(ABC):
    @abstractmethod
    async def transcribe(self, *args, **kwargs) -> Transcription:
        raise NotImplementedError


class WhisperTranscriptor(BaseTranscriptor):

    def __init__(
        self,
        stt_model: WhisperSTT,
    ):
        self.stt_model = stt_model

    async def transcribe(self, audio_file_path: str) -> Transcription:
        st = time.time()
        text = await self.stt_model.transcribe_audio(file_path=audio_file_path)
        transcription = Transcription(text=text)
        et = time.time()
        logger.info(f"Transcribed audio (time: {et-st:.3f}s.): {transcription}")
        return transcription
