import os
import json
import base64
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
import websockets

from logs import logger

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")


# You need to collect
# alignment info for audio post-processing
class ElevenLabsAlignmentInfo(BaseModel):
    chars: List[str]
    charStartTimesMs: List[int]
    charDurationsMs: List[int]


class ElevenLabsResponse(BaseModel):
    audio_data: bytes
    alignment_info: ElevenLabsAlignmentInfo
    output_format: str  # needed for post-processing


# Combines alignment info of whole audio into a single one
# NOTE: This function is optional. You can directly process streamed
# bytes chunks by just collecting audio data and
# bytes into ElevenLabsResponse
def combine_alignment_info(
    alignment_info: List[ElevenLabsAlignmentInfo],
) -> ElevenLabsAlignmentInfo:
    """
    Integrates multiple alignment info blocks
    into a single alignment info dictionary.
    """
    cumulative_run_time = 0
    chars: List[str] = []
    char_start_times_ms: List[int] = []
    char_durations_ms: List[int] = []
    for block in alignment_info:
        chars.extend([" "] + block.chars)
        char_durations_ms.extend([block.charDurationsMs[0]] + block.charDurationsMs)
        start_times = [0] + [
            time + cumulative_run_time for time in block.charStartTimesMs
        ]
        char_start_times_ms.extend(start_times)
        cumulative_run_time += sum(block.charDurationsMs)
    combined_info = ElevenLabsAlignmentInfo(
        chars=chars,
        charStartTimesMs=char_start_times_ms,
        charDurationsMs=char_durations_ms,
    )
    return combined_info


class ElevenLabsTTS:
    def __init__(
        self,
        model_id: str,
        voice_id: str,
        stability: float = 0.2,
        similarity_boost: float = 0.75,
        output_format: str = "mp3_44100",
    ):
        self.websockets_uri: str = (
            "wss://api.elevenlabs.io/v1/text-to-speech/"
            f"{voice_id}/stream-input?model_id={model_id}"
        )
        self.api_key: str = ELEVENLABS_API_KEY
        self.output_format: str = output_format
        self.stability: float = stability
        self.similarity_boost: float = similarity_boost

    async def generate_audio(
        self,
        text: str,
    ) -> ElevenLabsResponse:
        async with websockets.connect(self.websockets_uri) as websocket:
            logger.info("Start sending text to ElevenLabs")
            bos_message = {
                "text": " ",
                "voice_settings": {
                    "stability": self.stability,
                    "similarity_boost": self.similarity_boost,
                },
                "output_format": self.output_format,
                "xi_api_key": self.api_key,
            }
            await websocket.send(json.dumps(bos_message))

            input_message = {"text": text}
            await websocket.send(json.dumps(input_message))

            eos_message = {"text": ""}
            await websocket.send(json.dumps(eos_message))

            audio_chunks = []
            alignment_info = []
            while True:
                try:
                    response = await websocket.recv()
                    data = json.loads(response)
                    logger.info("Received response from elevenlabs")
                    if data.get("audio"):
                        logger.info(f"Received audio chunk: {len(data.get('audio'))}")
                        chunk = base64.b64decode(data.get("audio"))
                        audio_chunks.append(chunk)
                        if data.get("alignment"):
                            logger.info(f"Alignment info: {data.get('alignment')}")
                            alignment_info.append(
                                ElevenLabsAlignmentInfo(**data.get("alignment"))
                            )
                        else:
                            logger.error(f"No alignment info in response: {response}")
                            break
                    else:
                        logger.error(f"No audio data in the response: {response}")
                        break
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Connection closed")
                    break
        combined_alignment_info = combine_alignment_info(alignment_info)
        audio_data = b"".join(audio_chunks)
        response = ElevenLabsResponse(
            audio_data=audio_data,
            alignment_info=combined_alignment_info,
            output_format=self.output_format,
        )
        return response
