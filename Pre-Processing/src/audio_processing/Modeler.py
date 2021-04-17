import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import seaborn as sns
import os
#from fastai.vision.all import *

class Modeler:
    def __init__(self, data_path):
        self.data_path = data_path
        print(f'DATA_PATH: {self.data_path}')

    def build(self):

        """
        INNER FUNCTIONS
        """

        def load_data():
            with open(self.data_path, "r") as fp:
                data = json.load(fp)

            #convert list -> np.array()
            inputs = np.array(data["features"])

            #mms = np.array(data["mms"])
            labels = np.array(data["labels"])

            mms = np.array(data["mms"])
            mms = normalize_target(mms)

            print(f'labels.shape {labels.shape} labels: {labels}, mms.shape: {mms.shape} mms: {mms}')
            return inputs, labels

        def normalize_target(y):
            #sns.pairplot(train_dataset[[""]], diag_kind="kde")
            #train_stats = train_dataset.describe()
            #(x - train_stats['mean']) / train_stats['std']
            y = (y - np.mean(y, axis=0)) / np.std(y, axis=0)
            return y

        def prepare_datasets(test_size, validation_size):
            inputs, targets = load_data()


            print(f'inputs { inputs.shape} targets { targets.shape }')

            inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=test_size)
            inputs_train, inputs_validation, targets_train, targets_validation = train_test_split(inputs, targets, test_size=validation_size)

            inputs_train = inputs_train[..., np.newaxis] #4d array -> (num_samples, # frames, feautures, 1)
            inputs_validation = inputs_validation[..., np.newaxis] #4d array -> (num_samples, # frames, feautures, 1)
            inputs_test = inputs_test[..., np.newaxis] #4d array -> (num_samples, # frames, feautures, 1)

            return inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test

        def build_model_classifier(input_shape): #60%, 25%, 15%
            model = keras.Sequential()

            #layer 1
            model.add(keras.layers.Conv2D(
                32, (3,3), activation="relu", input_shape=input_shape
                #32, (3,3), activation=LeakyReLU(), input_shape=input_shape
            ))
            model.add(keras.layers.MaxPool2D(
                (3,3), strides=(2,2), padding="same"
            ))
            model.add(keras.layers.BatchNormalization())

            #layer 2
            model.add(keras.layers.Conv2D(
                32, (3,3), activation="relu", input_shape=input_shape
                #32, (3,3), activation=LeakyReLU(), input_shape=input_shape
            ))
            model.add(keras.layers.MaxPool2D(
                (3,3), strides=(2,2), padding="same"
            ))
            model.add(keras.layers.BatchNormalization())

            #layer 3
            model.add(keras.layers.Conv2D(
                32, (2,2), activation="relu", input_shape=input_shape
                #32, (3,3), activation=LeakyReLU(), input_shape=input_shape
            ))
            model.add(keras.layers.MaxPool2D(
                (2,2), strides=(2,2), padding="same"
            ))
            model.add(keras.layers.BatchNormalization())

            #flatten
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(64, activation="relu"))
            #model.add(keras.layers.Dense(64, activation=LeakyReLU()))
            model.add(keras.layers.Dropout(0.3))

            #output
            model.add(keras.layers.Dense(2, activation="softmax")) #CLASSIFIER
            #model.add(keras.layers.Dense(1))

            return model

        def build_model_regressor(input_shape): #60%, 25%, 15%
            model = keras.Sequential()

            #layer 1
            model.add(keras.layers.Conv2D(
                32, (3,3), activation="relu", input_shape=input_shape
                #32, (3,3), activation=LeakyReLU(), input_shape=input_shape
            ))
            model.add(keras.layers.MaxPool2D(
                (3,3), strides=(2,2), padding="same"
            ))
            model.add(keras.layers.BatchNormalization())

            #layer 2
            model.add(keras.layers.Conv2D(
                32, (3,3), activation="relu", input_shape=input_shape
                #32, (3,3), activation=LeakyReLU(), input_shape=input_shape
            ))
            model.add(keras.layers.MaxPool2D(
                (3,3), strides=(2,2), padding="same"
            ))
            model.add(keras.layers.BatchNormalization())

            #layer 3
            model.add(keras.layers.Conv2D(
                32, (2,2), activation="relu", input_shape=input_shape
                #32, (3,3), activation=LeakyReLU(), input_shape=input_shape
            ))
            model.add(keras.layers.MaxPool2D(
                (2,2), strides=(2,2), padding="same"
            ))
            model.add(keras.layers.BatchNormalization())

            #flatten
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(64, activation="relu"))
            #model.add(keras.layers.Dense(64, activation=LeakyReLU()))
            model.add(keras.layers.Dropout(0.3))

            #output
            #model.add(keras.layers.Dense(2, activation="softmax")) #CLASSIFIER
            model.add(keras.layers.Dense(1))

            return model

        """
        END
        """
        def assemble_classifier():
            #Build Procedure
            inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test = prepare_datasets(0.25, 0.2)

            input_shape = (inputs_train.shape[1], inputs_train.shape[2], inputs_train.shape[3])

            #print(f'inputs_test, targets_test: \n{inputs_test.shape}, {targets_test.shape}\ntrain:{inputs_train.shape} {targets_train.shape}')

            model = build_model_classifier(input_shape)

            #compile
            optimizer = keras.optimizers.Adam(learning_rate=0.0001) #ADAM
            #optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)

            model.compile(optimizer=optimizer,
                        loss="sparse_categorical_crossentropy",
                        #loss="mse",
                        metrics=["accuracy"]
                        #metrics=["mae", "mse"]
                        )
            model.summary()

            #train
            model.fit(inputs_train, targets_train,
                    validation_data=(inputs_validation, targets_validation),
                    epochs=30,
                    batch_size=4
                )

            model.summary()

            #evaluate
            test_error, test_accuracy = model.evaluate(inputs_test, targets_test, verbose=1)
            print(f'Accuracy on test set is {test_accuracy}\nError is {test_error}')

            model.save(os.path.abspath("../models/classifier.h5"), save_format='h5')
            print(f'Finished assmebling classifier...')

        def assemble_regressor():
            #Build Procedure
            inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test = prepare_datasets(0.25, 0.2)

            input_shape = (inputs_train.shape[1], inputs_train.shape[2], inputs_train.shape[3])

            #print(f'inputs_test, targets_test: \n{inputs_test.shape}, {targets_test.shape}\ntrain:{inputs_train.shape} {targets_train.shape}')

            model = build_model_regressor(input_shape)

            #compile
            #optimizer = keras.optimizers.Adam(learning_rate=0.0001) #ADAM
            optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)

            model.compile(optimizer=optimizer,
                        #loss="sparse_categorical_crossentropy",
                        loss="mse",
                        #metrics=["accuracy"]
                        metrics=["mae", "mse"]
                        )
            model.summary()

            #train
            model.fit(inputs_train, targets_train,
                    validation_data=(inputs_validation, targets_validation),
                    epochs=30,
                    batch_size=4
                )

            model.summary()

            #evaluate
            output = model.evaluate(inputs_test, targets_test, verbose=1)
            print(f'Accuracy on test set is {output})')
            #\nError is {test_error}')

            model.save(os.path.abspath("../models/regressor.h5"), save_format='h5')
            print(f'Finished assmebling regressor...')

        assemble_regressor()
