from typing import List, Literal, TypedDict
from pydantic import BaseModel


class ElevenLabsAlignmentInfo(BaseModel):
    chars: List[str]
    charStartTimesMs: List[int]
    charDurationsMs: List[int]


class ElevenLabsResponse(BaseModel):
    audio_data: bytes
    alignment_info: ElevenLabsAlignmentInfo
    output_format: str


class EmotionAnnotationSpan(BaseModel):
    start_time: int
    end_time: int


class AudioData(BaseModel):
    data: bytes


class PromptMessage(TypedDict):
    role: Literal["user", "system"]
    content: str


class Transcription(BaseModel):
    text: str


class UserQuery(BaseModel):
    query: str
    history: List[str]
