# -*- coding: utf-8 -*-

import cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
node = True
import os
import python_speech_features as mfcc

modelsPath = "models/" #path to our model file
source = 'trainingData/' #Path of the audio files we will use to create the model

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

# Extraction features for each speaker
features = np.asarray(()) #we created Array
sourceFolder = [os.path.join(name)
    for name in os.listdir(source)] #We got the folders in the TrainingData folder.
print("Source Folders: ",sourceFolder)
sources = [] #create a new list. We will take the .wav files in the folders in the training data folder into this list.
for x in sourceFolder:
    for name in os.listdir(source + x): #TrainingData/x where x is the folder in it. This function will work for each folder.
        if name.endswith('.wav'): #If it is a wav file in TrainingData/x;
            nn = "{}".format(x)+"/"+"{}".format(name) #Path
            sources.append(nn) #Adding it to our list.

for path in sources:    
    path = path.strip()   
    print(path)
    # Read the voice
    sr,audio = read(source + path)
    print(source + path)
    # Let's explain the 40-dimensional MFCC and delta MFCC properties
    vector   = extract(audio,sr)
    if features.size == 0: #If we doesn't have any data
        features = vector #Features will equal to vector and program ends.
    else: 
        features = np.vstack((features, vector)) #We stack arrays vertically (on a row basis) sequentially.
    if node == True:    
        gmm = GMM(n_components = 16, max_iter = 200,  covariance_type='diag',n_init = 3) #We are calling gmm function.
        gmm.fit(features)
        # We save the models we calculated to the folder
        picklefile = path.split("-")[0]+".gmm"
        cPickle.dump(gmm,open(modelsPath + picklefile,'w'))
        print "  >> Modeling complete for file: ",picklefile,' ',"| Data Point = ",features.shape    
        features = np.asarray(()) 
