from prefect import flow, task, serve

import os
import json
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from bson import ObjectId
from typing import List
import Config.Config as Config

# Define MongoDB connection parameters
username = Config.username
password = Config.pwd
host = Config.url
port = Config.port
database_name = Config.dbname
collection_name = Config.dbcollection

# Connect to MongoDB
client = AsyncIOMotorClient(f"mongodb://{username}:{password}@{host}:{port}")
db = client[database_name]
collection = db[collection_name]


# Define a Pydantic model for audio data
class AudioData(BaseModel):
    path: str
    sampling_rate: int
    array: List[float]


class Recording(BaseModel):
    sentence: str
    audio: AudioData


# Function to fetch data from MongoDB
@task
async def fetch_data():
    recordings = []
    async for document in collection.find({"$and": [{"sentence": {"$exists": True}}, {"in_progress": {"$exists": False}}]}):
        recordings.append({
            "sentence": document["sentence"],
            "audio": {
                "path": document["audio"]["path"],
                "sampling_rate": document["audio"]["sampling_rate"],
                "array": document["audio"]["array"]
            }
        })
    return recordings


# Function to process data and save it in the required format
@flow(log_prints=True)
async def build_audio_dataset_from_mongodb():
    # Fetch data from MongoDB
    data = await fetch_data()

    # Convert data to Datasets library format
    dataset = Dataset.from_pandas(pd.DataFrame(data))

    # Save dataset to '../Dataset' folder
    dataset.save_to_disk('../Dataset')


if __name__ == "__main__":
    build_audio_dataset_from_mongodb.serve(name="build_audio_dataset_from_mongodb",
                                   tags = ["medium", "DataLabeler", "DatasetBuilder"],
                                   description = "Create a DataSet for ASR",)