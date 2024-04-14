from openai import OpenAI
from flask import request 
from utils import similarity 
import numpy as np
import scipy.io.wavfile as wav
import json

def float_to_wav(byte_data, file_path, sample_rate=44100):
    
    with wav.Wave_write(file_path) as wf:
        wf.setnchannels(1) 
        wf.setsampwidth(2) 
        wf.setframerate(sample_rate)
        wf.writeframes(byte_data)

def evaluate2():
    client = OpenAI()
    audio = json.loads(request.form.get("audio"))
    float_to_wav(audio,"./upload/uploaded.wav")
    file = open("./upload/uploaded.wav","rb")
    transcript = client.audio.transcriptions.create(
      model="whisper-1", 
      file=file,
      response_format="verbose_json",
    )
    translate = client.audio.translations.create(
        model="whisper-1",
        file=file
    )
    return {"transcript":transcript.text,"confidence":2.718**(transcript.segments[0]["avg_logprob"])
,"translated": translate.text,"similarity":similarity(translate.text,request.form.get("text",""))},200