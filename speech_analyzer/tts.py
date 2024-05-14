import io
import time
from typing import List
from pydub import AudioSegment

from logs import logger
from schema import AudioData, EmotionAnnotationSpan, ElevenLabsAlignmentInfo
from model_clients import ElevenLabsTTS


class EmotionAnnotationTrimmer:
    def __init__(self, annotation_marker: str = "--"):
        self.annotation_marker = annotation_marker

    def process(
        self, audio_data: bytes, alignment_info: ElevenLabsAlignmentInfo
    ) -> AudioData:
        annotation_spans = self._get_annotation_spans(alignment_info)
        trimmed_audio = self._trim_emotion_annotations(
            audio_data=audio_data,
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
            if alignment_info.chars[i : i + marker_length] == self.annotation_marker:
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
        return spans

    @staticmethod
    def _trim_emotion_annotations(
        audio_data: bytes,
        annotation_spans: List[EmotionAnnotationSpan],
    ) -> bytes:
        st = time.time()
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
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
        self.emotion_annotation_remover = EmotionAnnotationTrimmer()

    async def generate(self, text: str) -> AudioData:
        logger.info(f"Start generating audio: {text}")
        st = time.time()
        response = await self.model.generate_audio(text=text)
        et = time.time()
        logger.info(
            f"Finish generating audio (time: {et - st:.3f} s.), "
            f"len: {len(response.audio_data)} bytes"
        )
        processed_audio = self.emotion_annotation_remover.process(
            audio_data=response.audio_data,
            alignment_info=response.alignment_info,
        )
        return processed_audio
