from google.cloud import speech
from flask import request
from googletrans import Translator  
from utils import similarity 

def evaluate():

    client = speech.SpeechClient()
    t = Translator(service_urls=["translate.google.co.in"])
    audio = request.files["audio"]
    audio_data = audio.read()
    audio = speech.RecognitionAudio(content=audio_data)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        enable_automatic_punctuation=True,
        audio_channel_count=1,
        enable_separate_recognition_per_channel=True,
        language_code=request.form.get("language","en-IN"),
    )
    response = client.recognize(request={"config": config, "audio": audio})
    transcript = response.results[0].alternatives[0].transcript
    translatedText = t.translate(transcript,dest="en",src=request.form.get("language","en")[:2])
    return {"transcript":transcript,"confidence":response.results[0].alternatives[0].confidence
,"translated": translatedText.text,"similarity":similarity(translatedText.text,request.form.get("text",""))},200