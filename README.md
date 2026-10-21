# Computerpreter Training
Forest Young and Victor Young

This repository holds the preprocessing and training code for Computerpreter. Computerpreter aims to optimize Deaf accessibility through real-time American Sign Language translation. Computerpreter consists of the fingerspelling classifier and the isolated sign classifier, divided into two different folders in this repository.

To learn more about Computerpreter, please visit [computerpreter.streamlit.app](https://computerpreter.streamlit.app) to test it out! To view the app combining Computerpreter's features into a fully functional Streamlit web app, please visit [Computerpreter's repository for Streamlit](https://github.com/FooStoch/ASL_Signs_Detection/).

View [our amazing slide deck](https://docs.google.com/presentation/d/192PEwEdJvzwmJIR5-nHPEJYOVw0Z5sMqbfh7zhuwz0A/edit?usp=sharing) for more details on community engagement, project development, entrepreneurship, and much more!

## Fingerspelling Classifier

For the fingerspelling classifier, we used Python, OpenCV and Mediapipe! First, we collected around 26000 images for handshapes A-Z using OpenCV and they are saved in `./data`. Second, we created a dataset using MediaPipe to extract 21 hand landmarks, totaling 63 features, for each image. Third, we coded 4 different ML architectures with 4 hyperparameter specifications each.

Our data is not uploaded here, but the code provides our simple and fast method for getting data. Simply run `collect_imgs.py` and let the webcam capture 1000 images of you holding up the requested handshape. Make sure to move and rotate your hand for higher variety as this better reflects real-life fingerspelling. These images will be dumped into the newly created `data` folder. Afterwards, running `create_dataset.py` will run these images through MediaPipe Hands to get the proper landmarks for each image, stored in `data.pickle`, which is uploaded here.

Please view our training progress in `train_model.ipynb`. You can view training history and specific hyperparameter specifications for our **MLP**, **SVM**, **XGB**, and **RF** models! The final test accuracies are present in `results.json`, and the best model (`MLP_3.pt`) can be found in the `models` folder. Note that this folder will be filled with all models after running `train_model.ipynb`.

The Python version is 3.9.13. Make sure to install all the requirements in `requirements.txt` and then run the program in `inference_classifier.py`.

## Sign Classifier

The dynamic sign classifier uses MediaPipe Holistic to extract 42 hand landmarks, 33 pose landmarks, and 30 face landmarks, and then predict one of the over 50 common signs trained. The code uses data from MS-ASL, WLASL 2000, and ASL Citizen, which are publically available for non-commercial purposes. MS-ASL and WLASL are provided because they only contain links and times to videos on the Internet, while ASL Citizen contains actual video files and is omitted for storage. Paste ASL Citizen's folder in the same directory as MS-ASL and WLASL. Many logs, indexes, CSVs, and pickle files aren't shared due to privacy and size.

Because MS-ASL and WLASL provide links to YouTube videos and other websites, we use `yt-dlp`. This requires `cookies.txt` to be present in `Dynamic Sign`, which can be gotten using this extension: [Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc). **Never share `cookies.txt` online!!!**

Because MediaPipe selects 468 face landmarks in total, `landmark_selection.ipynb` selects the most important 30. After preprocessing, 36 models are trained on the data. The models are 4 sets of 9 different architectures, 5 of which are self-made and the other 4 are research-inspired.

| Bi-LSTM + attention | Bi-GRU + attention | $$\color{blue}{\text{Residual temporal CNN}}$$ | Transformer encoder | TCN + Bi-GRU + attention |
| -------- | -------- | -------- | -------- | -------- |

| Temporal I3D (inception) | TGCN (temporal GCN) | ST-GCN (spatio-temporal GCN) | SL-GCN (decoupled GCN) |
| -------- | -------- | -------- | -------- |

In our training, the last 4 research-inspired models were trained on Google Colab using free compute units (YAY!). However, they did not outperform the **residual temporal CNN**, so `Computerpreter_4th_Iteration_train_model.ipynb` isn't actually needed for getting `best_model.pt`. It is still provided for your reference! Finally, test the classifier in `asl_detector.py`, which actually supports running all the different models.

Below is a list of all folders that are present or will be created upon running the code.

* `collected_data`: This folder is present and contains information on synonyms so that data from the three sources can be unified seamlessly.
* `raw_videos`: This folder will be created after running `video_downloader.py` and contains all the downloaded videos from WLASL and MS-ASL as well as formatted videos copied from ASL Citizen.
* `videos`: This folder will be created after running `preprocess.py` and contains temporally and spatially cropped videos.
* `landmarks`: This folder will be created after running `landmark_extraction.ipynb` and contains the all 543 landmarks for every frame in every video. These landmarks are temporally normalized (30 frames per video) and spatially normalized (based on shoulder-to-shoulder length).
* `selected_landmarks`: This folder will be created after running `landmark_selection.ipynb` and contains the top 30 face landmarks, totaling 105 landmarks, selected with an ExtraTreesClassifier.
* `models`: This folder will be created after running the training notebooks and contains the best model along with hyperparameters and training history. We provide it for your convenience in testing the model out!

## Thank you!

Thank you for your attention to our project, Computerpreter! We are very thankful for the opportunity for us to be able to create something so fun and meaningful! 

Through this challenge, we both learned so much. From researching the gap between Artificial Intelligence and American Sign Language, this incredible journey has been one of great significance and magnificence. We want to give a special thanks to Mrs. Tolley for giving us inspiration. If you have any questions about our project, feel free to reach out.
