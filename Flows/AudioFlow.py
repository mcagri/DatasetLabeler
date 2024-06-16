from transformers import pipeline
import torch

from prefect import flow, task, serve
from pymongo import MongoClient
import configparser
import pandas as pd
import requests
import json

import AudioUtils.VAD as Vad
import soundfile as sf
import os
import librosa
import webrtcvad

import Config.Config as Config

client = MongoClient(Config.url,
                         username=Config.username,
                         password=Config.pwd,
                         authSource="admin",
                         authMechanism='SCRAM-SHA-256')
db = client[Config.dbname]
collection = db[Config.dbcollection]


pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch.float16,
    device="cuda:0",
    model_kwargs= {"attn_implementation": "flash_attention_2"}
)

@task
def process_stereo2mono(path: str = "../Data/", mono_path: str = "../Data/mono/",
                        resample_path: str = "../Data/16khz/", sample_rate: int = 16000):
    sound_files = {}
    for filename in os.listdir(path):
        try:
            data, sr = sf.read(path + "/" + filename)
            # Separate channels
            left_channel = data[:, 0]  # Left channel
            right_channel = data[:, 1]  # Right channel
            sf.write(mono_path + "left_channel_" + filename, left_channel, sr)
            sf.write(mono_path + "right_channel_" + filename, right_channel, sr)
        except Exception as e:
            print(e)

    for filename in os.listdir(mono_path):
        try:
            y, sr = librosa.load(mono_path + filename, sr=sample_rate)
            sf.write(resample_path + filename, y, samplerate=sample_rate)
        except Exception as e:
            print(e)

@task
def process_vad_chunking(input_path: str = "../Data/16khz/", output_path: str = "../Data/chunks/", aggressiveness: int = 2):
    for filename in os.listdir(input_path):
        try:
            audio, sample_rate = Vad.read_wave(input_path + filename)
            vad = webrtcvad.Vad(aggressiveness)
            frames = Vad.frame_generator(30, audio, sample_rate)
            frames = list(frames)
            segments = Vad.vad_collector(sample_rate, 30, 300, vad, frames)
            for i, segment in enumerate(segments):
                path = 'chunk-%0003d-' % (i,) + filename
                print(' Writing %s' % (path,))
                Vad.write_wave(output_path + path, segment, sample_rate)
        except Exception as e:
            print(e)


@task
def flush_buffer(buffer:list):
    collection.insert_many(buffer)


@task
def transcribe(input_path: str = "../Data/chunks/", sample_rate: int = 16000, buffer_size: int = 100):
    buffer = []
    for filename in os.listdir(input_path):
        temp_path = os.path.join(input_path, filename)
        transcript = pipe(temp_path, generate_kwargs={"language": "tr"})["text"]
        data, sr = librosa.load(temp_path, sr=sample_rate, mono=True)
        temp_object = {'sentence':transcript,
                       'audio':{'path':temp_path,
                                'sampling_rate':sample_rate,
                                'array':data.tolist()}
                       }
        buffer.append(temp_object)
        if len(buffer) >= buffer_size:
            flush_buffer(buffer)
            buffer.clear()
    if len(buffer) > 0:
        flush_buffer(buffer)


@flow(log_prints=True)
def audio_ingestion_file_ops():
    process_stereo2mono(Config.path, Config.mono_path, Config.resample_path, Config.sample_rate)
    process_vad_chunking(Config.resample_path, Config.chunk_path, aggressiveness=2)
    transcribe(Config.chunk_path, Config.sample_rate, Config.buffer_size)


if __name__ == "__main__":
    audio_ingestion_file_ops.serve(name="audio_ingestion_file_ops",
                                   tags = ["medium", "DataLabeler"],
                                   description = "Create a DataSet for ASR",)
