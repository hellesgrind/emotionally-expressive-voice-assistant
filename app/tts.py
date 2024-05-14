import io
import time
from typing import List
from pydub import AudioSegment
from pydantic import BaseModel

from logs import logger
from elevenlabs_model import ElevenLabsAlignmentInfo, ElevenLabsResponse, ElevenLabsTTS


class EmotionAnnotationSpan(BaseModel):
    start_time: int
    end_time: int


class AudioData(BaseModel):
    data: bytes


class EmotionAnnotationTrimmer:
    def __init__(self, annotation_marker: str = "--"):
        self.annotation_marker = annotation_marker

    def process(
        self,
        elevenlabs_response: ElevenLabsResponse,
    ) -> AudioData:
        annotation_spans = self._get_annotation_spans(
            alignment_info=elevenlabs_response.alignment_info
        )
        trimmed_audio = self._trim_emotion_annotations(
            audio_data=elevenlabs_response.audio_data,
            audio_format=elevenlabs_response.output_format,
            annotation_spans=annotation_spans,
        )
        return AudioData(data=trimmed_audio)

    def _get_annotation_spans(
        self,
        alignment_info: ElevenLabsAlignmentInfo,
    ) -> List[EmotionAnnotationSpan]:
        marker_length: int = len(self.annotation_marker)
        marker_positions: List[int] = []
        for i in range(len(alignment_info.chars) - marker_length + 1):
            chars = "".join(alignment_info.chars[i : i + marker_length])
            if chars == self.annotation_marker:
                marker_positions.append(alignment_info.charStartTimesMs[i])
        if len(marker_positions) % 2 != 0:
            logger.error("Uneven number of emotion markers found.")
            return []

        spans = []
        for i in range(0, len(marker_positions), 2):
            spans.append(
                EmotionAnnotationSpan(
                    start_time=marker_positions[i], end_time=marker_positions[i + 1]
                ),
            )
        logger.info(f"Emotional span timelines: {spans}")
        return spans

    @staticmethod
    def _trim_emotion_annotations(
        audio_data: bytes,
        audio_format: str,
        annotation_spans: List[EmotionAnnotationSpan],
    ) -> bytes:
        st = time.time()
        if "mp3" in audio_format:
            pydub_format = "mp3"
        elif "mp4" in audio_format:
            pydub_format = "mp3"
        elif "wav" in audio_format:
            pydub_format = "wav"
        elif "pcm" in audio_format:
            pydub_format = "pcm"
        else:
            logger.error(f"Format {audio_format} not supported")
            return b""
        audio_segment = AudioSegment.from_file(
            file=io.BytesIO(audio_data),
            format=pydub_format,
        )
        parts = []
        previous_end = 0
        for span in annotation_spans:
            parts.append(audio_segment[previous_end : span.start_time])
            previous_end = span.end_time

        if previous_end < len(audio_segment):
            parts.append(audio_segment[previous_end:])

        final_audio = sum(parts, AudioSegment.silent(duration=0))
        buffer = io.BytesIO()
        final_audio.export(buffer, format="mp3")

        et = time.time()
        logger.info(
            f"Trimmed audio (time: {et - st:.3f} s.), "
            f"len: {len(buffer.getvalue())} bytes"
        )
        return buffer.getvalue()


class ElevenLabsSpeechGeneration:
    def __init__(
        self,
        model: ElevenLabsTTS,
    ):
        self.model = model
        self.emotion_annotation_trimmer = EmotionAnnotationTrimmer()

    async def generate(self, text: str) -> AudioData:
        logger.info(f"Start generating audio: {text}")
        st = time.time()
        response = await self.model.generate_audio(text=text)
        et = time.time()
        logger.info(
            f"Finish generating audio (time: {et - st:.3f} s.), "
            f"len: {len(response.audio_data)} bytes"
        )
        processed_audio = self.emotion_annotation_trimmer.process(
            elevenlabs_response=response,
        )
        return processed_audio
