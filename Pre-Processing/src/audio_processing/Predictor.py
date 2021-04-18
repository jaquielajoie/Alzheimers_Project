import json
import numpy as np
import tensorflow.keras as keras
import os
from audio_processing.AudioProcessor import AudioProcessor
from .TrainingLabeler import TrainingLabeler
import pandas as pd

#Train
#Epoch 30/30
#loss: 0.0811 - mae: 0.1950 - mse: 0.0811 - val_loss: 0.2647 - val_mae: 0.4692 - val_mse: 0.2647

#TEST METRICS
#LOSS, MAE, MSE
#Accuracy on test set is [0.05875355005264282, 0.15116603672504425, 0.05875355005264282])

class Predictor:
    def __init__(self, model_path, file_path, f, filename):
        self.model_path = model_path
        self.file_path = file_path
        self.filename = filename
        self.f = f

    def load_data(self, data_path):
        with open(data_path, "r") as fp:
            data = json.load(fp)

        #convert list -> np.array()
        inputs = np.array(data["features"])

        #mms = np.array(data["mms"])
        labels = np.array(data["labels"])

        #mms = np.array(data["mms"])
        #mms = normalize_target(mms)

        #print(f'labels.shape {labels.shape} labels: {labels}, mms.shape: {mms.shape} mms: {mms}')
        return inputs, labels


    def pre_process(self):
        processor = AudioProcessor(data_path='')
        training_labeler = TrainingLabeler(os.path.abspath("audio_processing/json/predict.json"))
        processor.get_features_from_file_path(training_labeler=training_labeler, file_path=self.file_path, filename=self.f, mms_df=None, f=self.filename, i=1)
        #just use this to pull one file
        training_labeler.save()
#Test Predictions: [[0.24143995]]
    def predict(self):
        self.pre_process()
        #load model
        model = keras.models.load_model(os.path.abspath('../models/regressor.h5'))

        inputs, targets = self.load_data(data_path=os.path.abspath("audio_processing/json/predict.json"))

        inputs = inputs[..., np.newaxis] #4d array -> (num_samples, # frames, feautures, 1)

        test_predictions = model.predict(inputs)


        print(f'Test Predictions: {test_predictions}')
