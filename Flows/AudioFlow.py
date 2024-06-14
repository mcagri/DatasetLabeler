from prefect import flow, task, serve
from pymongo import MongoClient
import configparser
import pandas as pd
import requests
import json

import AudioUtils.VAD as vad
import soundfile as sf
import os
import librosa

config = configparser.ConfigParser()
config.read("../Config/db.config")
username = config['MONGODB']['MONGO_USER']
pwd = config['MONGODB']['MONGO_PASSWORD']
dbname = config['MONGODB']['MONGO_DB']
url = config['MONGODB']['MONGO_URL']
port = config['MONGODB']['MONGO_PORT']
client = MongoClient(url,
                         username=username,
                         password=pwd,
                         authSource="admin",
                         authMechanism='SCRAM-SHA-256')
db = client[dbname]
collection = db['Audio']

@task
def process_stereo2mono(path: str = "./Data", mono_path: str = "./Data/16khz", sample_rate: int = 16000):
    sound_files = {}
    for filename in os.listdir(path):
        try:
            data, samplerate = sf.read(path + "/" + filename)
            # Separate channels
            left_channel = data[:, 0]  # Left channel
            right_channel = data[:, 1]  # Right channel
            left_channel, sr = librosa.load(left_channel, sr=sample_rate)
            right_channel, sr = librosa.load(right_channel, sr=sample_rate)
            sf.write(mono_path + "/" + "left_channel_" + filename, left_channel, sample_rate)
            sf.write(mono_path + "/" + "right_channel_" + filename, right_channel, sample_rate)
        except Exception as e:
            print(e)