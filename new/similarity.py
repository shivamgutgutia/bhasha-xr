import librosa
import numpy as np

def align_audio(audio1, audio2):
    xcorr = np.correlate(audio1, audio2, mode='full')
    delay = np.argmax(xcorr) - len(audio1) + 1

    if delay > 0:
        audio2_aligned = np.pad(audio2[:-delay], (delay, 0), mode='constant')
        audio1_aligned = audio1
    else:
        audio1_aligned = np.pad(audio1[:delay], (-delay, 0), mode='constant')
        audio2_aligned = audio2
    
    return audio1_aligned, audio2_aligned

def compare_audio(audio1_path, audio2_path):

    y1, sr1 = librosa.load(audio1_path)
    y2, sr2 = librosa.load(audio2_path)

    min_len = min(len(y1), len(y2))
    y1 = y1[:min_len]
    y2 = y2[:min_len]

    y1_aligned, y2_aligned = align_audio(y1, y2)

    mfcc1 = librosa.feature.mfcc(y=y1_aligned, sr=sr1)
    mfcc2 = librosa.feature.mfcc(y=y2_aligned, sr=sr2)

    dist = np.linalg.norm(mfcc1 - mfcc2, ord=1)

    similarity_score = 1 / (1 + dist)

    return similarity_score

audio1_path = "audio1.wav"
audio2_path = "audio2.wav"
similarity_score = compare_audio(audio1_path, audio2_path)
print("Similarity score:", similarity_score)
