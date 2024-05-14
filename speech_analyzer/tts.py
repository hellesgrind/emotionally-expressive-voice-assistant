import io
import time
from typing import List
from pydub import AudioSegment

from logs import logger
from schema import GeneratedAudio, EmotionAnnotationSpan, ElevenLabsAlignmentInfo
from model_clients import ElevenLabsTTS


class ElevenLabsSpeechGeneration:
    def __init__(
        self,
        model: ElevenLabsTTS,
    ):
        self.model = model

    async def generate(self, text: str) -> GeneratedAudio:
        logger.info(f"Start generating audio: {text}")
        st = time.time()
        response = await self.model.generate_audio(text=text)
        et = time.time()
        logger.info(
            f"Finish generating audio (time: {et - st:.3f} s.), "
            f"len: {len(response.audio_data)} bytes"
        )
        emotion_annotation_spans = self.get_annotation_spans(response.alignment_info)
        trimmed_audio = self.trim_emotion_annotations(
            audio_data=response.audio_data,
            emotion_placeholder_spans=emotion_annotation_spans,
        )
        return GeneratedAudio(audio_data=trimmed_audio)

    @staticmethod
    def get_annotation_spans(
        alignment_info: ElevenLabsAlignmentInfo,
    ) -> List[EmotionAnnotationSpan]:
        markers: List[int] = []
        spans: List[EmotionAnnotationSpan] = []
        emotion_placeholder_pattern: str = "--"  # TODO: Remove hardcode
        length_placeholder = len("--")

        for i in range(len(alignment_info.chars) - length_placeholder + 1):
            if all(
                alignment_info.chars[i + j] == emotion_placeholder_pattern[j]
                for j in range(length_placeholder)
            ):
                markers.append(alignment_info.charStartTimesMs[i])

        if len(markers) % 2 != 0:
            logger.error("Uneven number of emotion markers found.")
            return spans

        for i in range(0, len(markers), 2):
            spans.append(
                EmotionAnnotationSpan(start_time=markers[i], end_time=markers[i + 1])
            )
        return spans

    # TODO: There's no async PyDub, find a replacement
    @staticmethod
    def trim_emotion_annotations(
        audio_data: bytes,
        emotion_placeholder_spans: List[EmotionAnnotationSpan],
    ) -> bytes:
        st = time.time()
        audio_segment = AudioSegment.from_file(
            io.BytesIO(audio_data), format="mp3"  # TODO: Remove hardcode
        )
        markers: List[int] = []
        for span in emotion_placeholder_spans:
            markers.append(span.start_time)
            markers.append(span.end_time)
        # Process segments outside the exclusion zones
        parts = []
        previous_end = 0  # Start of the audio
        for i in range(0, len(markers), 2):
            exclude_start = markers[i]
            exclude_end = markers[i + 1]
            parts.append(audio_segment[previous_end:exclude_start])
            previous_end = exclude_end

        # Append the last segment if the last marker does not reach the end of the audio
        if previous_end < len(audio_segment):
            parts.append(audio_segment[previous_end:])

        # Combine the segments that are not excluded
        final_audio = sum(parts, AudioSegment.silent(duration=0))

        # Export the modified audio back to bytes
        buffer = io.BytesIO()
        final_audio.export(buffer, format="mp3")
        et = time.time()
        logger.info(
            f"Trimmed audio (time: {et - st:.3f} s.), "
            f"len: {len(buffer.getvalue())} bytes"
        )
        return buffer.getvalue()
