import pandas as pd
import numpy as np

import librosa
import soundfile

import os, glob, pickle

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def feature_ext(file, mfcc, chroma, mel):
    with soundfile.SoundFile(file) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result
    
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
    
}

observed_emotions=['calm', 'happy','sad','angry','fearful']


def load_data(a):
    x,y=[],[]
    for file in glob.glob("/home/ubuntu/environment/Audio_Files/Actor_*/*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
#         print(emotion)
        if emotion not in observed_emotions:
            continue
#         print(emotion)
        feature=feature_ext(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=a, random_state=0)
    
x_train,x_test,y_train,y_test=load_data(0.25)

print((x_train.shape[0], x_test.shape[0]))

print(f'Features extracted: {x_train.shape[1]}')

model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=50)

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))


def load_sampdata():
    x=[]
    for file in glob.glob("/home/ubuntu/environment/Sample_Audio_file/*.wav"):
        feature=feature_ext(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
    return np.array(x)
    
feature = load_sampdata()

y_samppred=model.predict(feature)

print("Emotion Of the Audio File: ",y_samppred)