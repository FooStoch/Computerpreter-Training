# Computerpreter-Training
Forest Young and Victor Young

This repository holds the preprocessing and training code for Computerpreter. Computerpreter aims to optimize Deaf accessibility through real-time American Sign Language translation. Computerpreter consists of the fingerspelling classifier and the isolated sign classifier, divided into two different folders in this repository.

To learn more about Computerpreter, please visit computerpreter.streamlit.app to test it out! To view the app combining Computerpreter's features into a fully functional Streamlit web app, please visit [Computerpreter's repository for Streamlit](https://github.com/FooStoch/ASL_Signs_Detection/).

To run all code, please install the dependencies in `requirements.txt`.

## Fingerspelling Classifier

The fingerspelling classifier uses MediaPipe Hands to extract 21 hand landmarks from each hand, and then predict one of the 26 handshapes. To obtain high-quality data, run the `GARBANZO`, which opens up the webcam and records 1000 frames for each letter. For high variation and model robustness, move and rotate your hand to mirror realistic fingerspelling. After MediaPipe preprocessing, 16 models are trained on the data. The models are 4 sets of 4 different architectures with different hyperparameters. Finally, test the classifier in `TRIBANZO`.

## Sign Classifier

The dynamic sign classifier uses MediaPipe Holistic to extract 42 hand landmarks, 33 pose landmarks, and 30 face landmarks, and then predict one of the over 50 common signs trained. The code uses data from MS-ASL, WLASL 2000, and ASL Citizen, which are publically available for non-commercial purposes. MS-ASL and WLASL are provided because they only contain links and times to videos on the Internet, while ASL Citizen contains actual video files and is omitted for storage. Paste ASL Citizen's folder in the same directory as MS-ASL and WLASL.

Because MediaPipe selects 468 face landmarks in total, `landmark_selection.ipynb` selects the most important 30. After preprocessing, 36 models are trained on the data. The models are 4 sets of 9 different architectures, 5 of which are self-made and the other 4 are research-inspired. Finally, test the classifier in `asl_detector.py`.

Because MS-ASL and WLASL provide links to YouTube videos and other websites, we use `yt-dlp`. This requires `cookies.txt` to be present in `Dynamic Sign`, which can be gotten using this extension: [Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc). **Never share `cookies.txt` online!!!**

Below is a list of all folders that are present or will be created upon running the code.

* `collected_data`: This folder is present and contains information on synonyms so that data from the three sources can be unified seamlessly.
* `raw_videos`: This folder will be created after running `video_downloader.py` and contains all the downloaded videos from WLASL and MS-ASL as well as formatted videos copied from ASL Citizen.
* `videos`: This folder will be created after running `preprocess.py` and contains temporally and spatially cropped videos.
* `landmarks`: This folder will be created after running `landmark_extraction.ipynb` and contains the all 543 landmarks for every frame in every video. These landmarks are temporally normalized (30 frames per video) and spatially normalized (based on shoulder-to-shoulder length).
* `selected_landmarks`: This folder will be created after running `landmark_selection.ipynb` and contains the top 30 face landmarks, totaling 105 landmarks, selected with an ExtraTreesClassifier. This folder is provided for convenient testing of the model.
