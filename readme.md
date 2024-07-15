# Emotion Detection Model

hello, I'm Rohit Jangir. this is my emotion detection project
This project involves building and testing an emotion detection model using the FER2013 dataset. The following instructions will guide you through setting up and running the project.

## Setup Instructions

### Step 1: Install Dependencies

Make sure you have Python installed on your system. Then, install the necessary Python modules by running the following command in your terminal:
```bash
pip install -r requirements.txt
```
 

### Step 2: Download and Prepare the Dataset

Create a folder named "data" in the same directory as your project files.
Download the FER2013 dataset from Kaggle.
this is the link of the dataset "[dataset](https://www.kaggle.com/datasets/msambare/fer2013)"
Extract the downloaded dataset into the "data" folder.

### Step 3: Training the Model (Optional)
If you want to train the model yourself, run the following command in your terminal:
```bash
"python TrainEmotionDetector.py"
```

### Step 4: Testing the Model
To test the pre-trained model, run the following command in your terminal:
```bash
"python TestEmotionDetector.py"
```

the model folder containes the trained model by me 


## Files Description
requirements.txt: Contains all the necessary modules for the project.
TrainEmotionDetector.py: Script to train the emotion detection model.
TestEmotionDetector.py: Script to test the emotion detection model.
data/: Directory where the FER2013 dataset should be extracted.
model/: trained model by me



# Notes
Ensure all files are in the same directory for the scripts to work correctly.
The dataset should be correctly extracted in the data folder for the training and testing scripts to function properly.
