from openai import OpenAI
from flask import request 
from utils import similarity 

def evaluate():
    client = OpenAI()
    #t = Translator(service_urls=["translate.google.co.in"])
    audio = request.files["audio"]
    audio.save("./upload/uploaded.wav")
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