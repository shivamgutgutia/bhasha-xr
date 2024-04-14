from openai import OpenAI
from flask import request 
from utils import similarity 
import numpy as np
import scipy.io.wavfile as wav

def float_to_wav(float_data, file_path, sample_rate=44100):
    # Normalize float data to the range [-1, 1]
    normalized_data = np.int16(float_data / np.max(np.abs(float_data)) * 32767)
    
    # Write the data to a WAV file
    with wav.Wave_write(file_path) as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 2 bytes (16 bits) per sample
        wf.setframerate(sample_rate)
        wf.writeframes(normalized_data)

def evaluate2():
    client = OpenAI()
    #t = Translator(service_urls=["translate.google.co.in"])
    audio = request.form.get("audio")
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