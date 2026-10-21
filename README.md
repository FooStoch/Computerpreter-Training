# Computerpreter-Training
This repository holds the preprocessing and training code for Computerpreter. Computerpreter aims to optimize Deaf accessibility through real-time American Sign Language translation. Computerpreter consists of the fingerspelling classifier and the isolated sign classifier, divided into two different folders in this repository.

To learn more about Computerpreter, please visit computerpreter.streamlit.app to test it out! To view the app combining Computerpreter's features into a fully functional Streamlit web app, please visit GGGGGGGGGGGGGGGGG.

To run all code, please install the dependencies in `requirements.txt`.

## Fingerspelling Classifier

The fingerspelling classifier uses MediaPipe Hands to extract 21 hand landmarks from each hand, and then predict one of the 26 handshapes. To obtain high-quality data, run the `GARBANZO`, which opens up the webcam and records 1000 frames for each letter. For high variation and model robustness, move and rotate your hand to mirror realistic fingerspelling. After MediaPipe preprocessing, 16 models are trained on the data. The models are 4 sets of 4 different architectures with different hyperparameters. Finally, test the classifier in `TRIBANZO`.

## Sign Classifier

The dynamic sign classifier uses MediaPipe Holistic to extract 42 hand landmarks, 33 pose landmarks, and 30 face landmarks, and then predict one of the over 50 common signs trained. The code uses data from MS-ASL, WLASL 2000, and ASL Citizen, which are publically available for non-commercial purposes. They are omitted for storage. Because MediaPipe selects 468 face landmarks in total, `weruhwiuhrciwuh` selects the most important 30. After preprocessing, 36 models are trained on the data. The models are 4 sets of 9 different architectures, 5 of which are self-made and the other 4 are research-inspired. Finally, test the classifier in `WOIERJOWIEJROWIEJR`.
