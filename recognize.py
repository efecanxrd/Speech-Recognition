# -*- coding: utf-8 -*-

import cPickle
import numpy as np
import warnings
from scipy.io.wavfile import read
from sklearn import preprocessing
warnings.filterwarnings("ignore")
import time
import os
import python_speech_features as mfcc

error = 0
samples = 0.0

# The folder with the audio files we recorded with Record.py or the audio files we want to know.
source = "Data/"
# The folder of the sound models we have trained
modelpath = "models/"

def calculate(array): #I don't need to explain more basic array and things
    rows,cols = array.shape 
    deltas = np.zeros((rows,20)) 
    N = 2
    for i in range(rows):
        index = [] 
        j = 1
        while j <= N:
            if i-j < 0:
              first =0
            else:
              first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

def extract(audio,rate): #Our function to extract the attribute of the audio.
    mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True) #We called our mfcc function from the #python_speech_features module. And we added diagnostic features
    mfcc_feature = preprocessing.scale(mfcc_feature) #preprocessing The package contains several common helper functions and substitution of transformer classes for a representative raw feature vectors that are more suitable for prediction.
    delta = calculate(mfcc_feature) #calculate_delta We calculate the variable we specified with mfcc.
    combined = np.hstack((mfcc_feature,delta)) #Sort arrays horizontally (as columns).
    return combined

gmmModels = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')] #Model files ending with .gmm in models
models = [cPickle.load(open(fname,'r')) for fname in gmmModels] #Opening model files
person = [fname.split("/")[-1].split(".gmm")[0] for fname in gmmModels] #Split and get the name of the person.

print "If you want to detect only one audio file, type 1\nIf you want to detect all audio files, type 0"
take = int(raw_input().strip())

if take == 1: 
	print "Enter the name of the file you want to define without '.wav': "
	path = raw_input().strip()  
	path = path + '.wav' 
	print "====================================\n= Checking the file: ", path
  #Read the voice
	sr,audio = read(source + path) 
	# Let's extract 40 dimensional MFCC and delta MFCC properties
	vector   = extract(audio,sr)
	log = np.zeros(len(models)) 
	for i in range(len(models)):
		gmm    = models[i]  #It is checked one by one with each model.
		scores = np.array(gmm.score(vector))
		log[i] = scores.sum()
	winner = np.argmax(log) #We rotate indexes of maximum values along our axis
	print "= >> Detected as person: "+person[winner], " "
	time.sleep(1.0)

elif take == 0:
	sources = [os.path.join(name) 
	for name in os.listdir(source) 
		if name.endswith('.wav')] #It takes all the files ending with .wav in the data folder in a list and detects them all.
	print(sources)
	# Let's read the data directory and get the audio files from the list
	for path in sources:   
		samples += 1.0
		path = path.strip()   
		print "====================================\n= Checking this file: ", path
		#Read the voice
		sr,audio = read(source + path)
		# Let's extract 40 dimensional MFCC and delta MFCC properties
		vector   = extract(audio,sr)
		log = np.zeros(len(models)) 
		for i in range(len(models)):
			gmm = models[i] #It is checked one by one with each model.
			scores = np.array(gmm.score(vector))
			log[i] = scores.sum()
		winner = np.argmax(log)
		print "= >> Detected as person: "+person[winner], " "
		checker_name = path.split("_")[0]
		if person[winner] != checker_name:
			error += 1
		time.sleep(1.0)

	percent = ((samples - error) / samples) * 100

	print("Percent for current test Performance with MFCC + GMM: ", percent,"%")
print "Voila!"
