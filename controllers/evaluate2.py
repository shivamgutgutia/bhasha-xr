from openai import OpenAI
from flask import request 
from utils import similarity 
import numpy as np
import wave

def float_to_wav(byte_data, file_path, sample_rate=44100):
    
    with wave.open(file_path,"wb") as wf:
        wf.setnchannels(1) 
        wf.setsampwidth(2) 
        wf.setframerate(sample_rate)
        wf.writeframes(byte_data.encode())

def evaluate2():
    client = OpenAI()
    audio = request.form.get("audio")
    print(audio)
    float_to_wav(byte_data=audio,file_path="upload/uploaded.wav")
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