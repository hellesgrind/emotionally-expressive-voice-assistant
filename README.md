# Emotional ElevenLabs Demo
## Getting Started 🌟
### Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.6 or higher
- Docker

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/hellesgrind/emotional-elevenlabs-demo.git
   ```

2. Navigate to the project directory:
   ```sh
   cd emotional-elevenlabs-demo
   ```

3. Add api keys to `.env`.

4. Build and run Docker container with app
   ```sh
   docker compose -f docker-compose.yml up -d --build 
   ```
   
## Usage 🎉

### Run Demo
1. Install client dependencies: `pip install -r requirements-client.txt`
2. Run client: `python main.py` 


## Project Structure

LLMScrapper is organized into several key components:

- `client.py`: Starts recording of audio and sends it to service.
- `speech_analyzer/main.py`: FastAPI app with websocket endpoint that audio data and returns audio data of generated response.
- `speech_analyzer/generation.py`: Prompt is initialized here. LLM generation part. Basically sends request to OpenAI API.
- `speech_analyzer/tts.py`: Post-processing of audio is here. Generation of audio too, actually.
