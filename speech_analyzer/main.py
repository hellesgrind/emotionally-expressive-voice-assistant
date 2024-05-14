from pydub import AudioSegment
from typing import List
from fastapi import FastAPI, WebSocket

from logs import logger
from stt import WhisperTranscriptor, WhisperSTT
from ttt import TextGeneration, OpenAIModel, UserQuery
from tts import ElevenLabsSpeechGeneration
from elevenlabs_model import ElevenLabsTTS


app = FastAPI()

# Initializing models
WHISPER_MODEL = WhisperSTT(model="whisper-1", language="en")
GENERATION_MODEL = OpenAIModel(model_name="gpt-4o")
TTS_MODEL = ElevenLabsTTS(
    model_id="eleven_monolingual_v1",
    voice_id="TxGEqnHWrfWFTfGW9XjX",
    stability=0.25,
    similarity_boost=0.75,
    output_format="mp3_44100",
)

# Initializing pipelines
transcriptor = WhisperTranscriptor(stt_model=WHISPER_MODEL)
text_generation = TextGeneration(model=GENERATION_MODEL)
speech_generation = ElevenLabsSpeechGeneration(model=TTS_MODEL)


dialog_history: List[str] = []


@app.websocket("/ws")
async def voice_chat(websocket: WebSocket):
    await websocket.accept()
    logger.info("Received websocket")
    file_path = "received_audio.wav"
    while True:
        try:
            data = await websocket.receive_bytes()
            logger.info(f"Received data: {len(data)}")
            audio_segment = AudioSegment(
                data=data, sample_width=2, frame_rate=44100, channels=1
            )
            audio_segment.export(file_path, format="wav")
            transcription = await transcriptor.transcribe(audio_file_path=file_path)
            response = await text_generation.generate(
                query=UserQuery(query=transcription.text, history=dialog_history)
            )
            audio_data = await speech_generation.generate(text=response)
            dialog_history.extend([f"User: {transcription.text}", f"You: {response}"])
            if len(dialog_history) >= 10:
                dialog_history.pop(0)
                dialog_history.pop(0)
            await websocket.send_bytes(audio_data.data)
        except Exception as e:
            logger.info(f"Websocket disconnect: {e}")
            break
