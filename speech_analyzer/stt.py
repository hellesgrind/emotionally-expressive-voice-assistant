import time
from abc import ABC, abstractmethod

from logs import logger
from model_clients import (
    HumeBatchAudio,
    WhisperSTT,
)
from schema import Transcription


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


class HumeBatchTranscriptor(BaseTranscriptor):
    def __init__(
        self,
        stt_model: HumeBatchAudio,
    ):
        self.stt_model: HumeBatchAudio = stt_model

    async def transcribe(self, audio_file_path: str) -> Transcription:
        st = time.time()
        transcription = self.stt_model.process_audio(filepaths=[audio_file_path])
        et = time.time()
        logger.info(f"Transcribed audio (time: {et-st:.3f}s.): {transcription}")
        return transcription
