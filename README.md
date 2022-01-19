# Speech Recognition
![EfecanLogo](https://avatars.githubusercontent.com/u/66366306?s=100&u=dc5e6f5b4a05d07958d9a867b803760aa2b1613e&v=4)
### A project with deep learning networks that recognizes who owns a voice using libraries like scipy, sklearn, and python-speech-features
![XhW](https://i.imgur.com/qHAcfhX.gif)
## Setup This Project
### Install Python2.7
- I recommend that you install Anaconda and install python 2.7 from the environments part of anaconda.
- After this installation, you can run the project by going to the project directory in your terminal and typing ```conda activate python2x```
### Install Libraries
- Switch to **Python2x** environment by typing ```conda activate Python2x```
- Then you can install the modules by typing ```pip install -r requirements.txt``` in the terminal. 
- Since you are running the project around the **Python2x** environment, you must also enable **Python2x** for use **pip** command
## How this is working?
A program that recognizes the sound of the specified file using models using methods such as mfcc gmm. **Code comments were entered as # comments on each line.**
- train.py : Used for model audio files in trainData folder
- recognize.py : It is used to define a selected file or all files in the data folder.
- requirements.txt : Text file containing the necessary libraries
- ./models : The folder where train.py outputs and recognize.py uses. Here are the models of the trained audio files.
- ./Data : The files here use recognize.py. Here you should drop the file you want to define .
- ./trainingData : Sound files to be used for modeling are placed here. 

For each audio file, it should be recorded as **VoiceName-Integer/VoiceFile.wav** Example: **Melissa-005/Melissa.wav** | Since the folder name is split, it should be saved like this.
