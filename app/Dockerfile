FROM python:3.11

WORKDIR app

COPY app/requirements.txt requirements.txt

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

RUN apt-get update \
    && apt-get install -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY app .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]