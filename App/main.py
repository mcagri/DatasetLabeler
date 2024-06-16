from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from bson import ObjectId
from typing import Dict
import Config.Config as Config
import soundfile as sf
import uvicorn
import io
import numpy as np

app = FastAPI()

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
client = AsyncIOMotorClient(f"mongodb://{Config.username}:{Config.pwd}@{Config.url}:{Config.port}")
db = client[Config.dbname]
collection = db[Config.dbcollection]


# Pydantic models
class AudioData(BaseModel):
    path: str
    sampling_rate: int
    array: list


class Recording(BaseModel):
    asr_sentence: str
    audio: AudioData


class UpdateRecording(BaseModel):
    sentence: str


# HTML Template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Audio Recordings</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <h1 class="mb-4">Audio Recordings</h1>
        <div id="recording-card" class="card">
            <div class="card-body">
                <h5 class="card-title">ASR Sentence</h5>
                <p id="asr-sentence" class="card-text"></p>
                <audio id="audio-player" controls class="w-100 mb-3">
                    <source id="audio-source" src="" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
                <div class="mb-3">
                    <label for="correct-sentence" class="form-label">Correct Sentence</label>
                    <input type="text" id="correct-sentence" class="form-control">
                </div>
                <button id="submit-btn" class="btn btn-primary">Submit</button>
                <button id="skip-btn" class="btn btn-secondary">Skip</button>
            </div>
        </div>
    </div>

    <script>
        async function fetchRecording() {
            const response = await fetch('/recording');
            if (response.ok) {
                const data = await response.json();
                document.getElementById('asr-sentence').innerText = data.asr_sentence;
                document.getElementById('audio-source').src = `/audio/${data._id}`;
                document.getElementById('audio-player').load();
                document.getElementById('correct-sentence').value = '';
                document.getElementById('recording-card').dataset.id = data._id;
            } else {
                alert('No more recordings to process.');
            }
        }

        async function submitRecording() {
            const correctSentence = document.getElementById('correct-sentence').value;
            const recordingId = document.getElementById('recording-card').dataset.id;
            if (correctSentence && recordingId) {
                const response = await fetch(`/recording/${recordingId}`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ sentence: correctSentence }),
                });
                if (response.ok) {
                    fetchRecording();
                } else {
                    alert('Failed to update recording.');
                }
            } else {
                alert('Please enter the correct sentence.');
            }
        }

        document.getElementById('submit-btn').addEventListener('click', submitRecording);
        document.getElementById('skip-btn').addEventListener('click', fetchRecording);

        window.onload = fetchRecording;
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    return html_template


@app.get("/recording", response_model=Dict)
async def get_recording():
    recording = await collection.find_one_and_update(
        {"sentence": {"$exists": False}, "in_progress": {"$exists": False}},
        {"$set": {"in_progress": True}}
    )
    if recording:
        recording["_id"] = str(recording["_id"])
        return recording
    else:
        raise HTTPException(status_code=404, detail="No more recordings to process")


@app.put("/recording/{recording_id}")
async def update_recording(recording_id: str, update: UpdateRecording):
    result = await collection.update_one(
        {"_id": ObjectId(recording_id)},
        {"$set": {"sentence": update.sentence}, "$unset": {"in_progress": ""}}
    )
    if result.modified_count == 1:
        return {"status": "success"}
    else:
        raise HTTPException(status_code=404, detail="Recording not found")


@app.get("/audio/{recording_id}")
async def get_audio(recording_id: str):
    recording = await collection.find_one({"_id": ObjectId(recording_id)})
    if recording:
        audio_array = np.array(recording['audio']['array'])
        sampling_rate = recording['audio']['sampling_rate']

        # Convert to wav-like bytes
        wav_io = io.BytesIO()
        sf.write(wav_io, audio_array, sampling_rate, format='WAV')
        wav_io.seek(0)

        return StreamingResponse(wav_io, media_type="audio/wav")
    else:
        raise HTTPException(status_code=404, detail="Audio not found")


@app.on_event("startup")
async def startup_db_client():
    client.admin.command('ping')


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)