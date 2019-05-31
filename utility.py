from keras import backend as K
import random
from keras.utils import np_utils
from keras.optimizers import Adamax
import numpy as np
import os
import pickle
from model import DFNet
import parameters


def load_data(dir, name):
    with open(dir + name, 'rb') as handle:
        data = np.array(pickle.load(handle, encoding='bytes'))
        print('{} loaded.'.format(name))
    return data


def load_dataset(dir, *args):
    my_list = []
    print('loading dataset .... ')
    for arg in args:
        my_list.append(load_data(dir, arg))
    print("Data dimensions:")
    for key, value in enumerate(args):
        print('{} SHAPE: {} '.format(value[:-4], my_list[key].shape))
    return tuple(my_list)


def save_data(data, dir, name):
    with open(dir + name, 'wb') as handle:
        pickle.dump(data, handle, protocol=4)


def make_adv_example():
    pass


def build_model():
    # Building  model
    print("Building and training DF model")

    model = DFNet.build(input_shape=parameters.INPUT_SHAPE, classes=parameters.NB_CLASSES)

    model.compile(loss="categorical_crossentropy", optimizer=parameters.OPTIMIZER,
                  metrics=["accuracy"])
    print("Model compiled")
    return model


def train_model(model, save_dir, X_train, X_valid, y_train, y_valid):
    description = "Training DF model for closed-world scenario on non-defended dataset"
    print(description)
    # Training the DF model
    print("Number of Epoch: ", parameters.NB_EPOCH)

    # Please refer to the dataset format in readme
    K.set_image_dim_ordering("tf")  # tf is tensorflow

    # Convert data as float32 type
    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    y_train = y_train.astype('float32')
    y_valid = y_valid.astype('float32')

    # we need a [Length x 1] x n shape as input to the DF CNN (Tensorflow)
    X_train = X_train[:, :, np.newaxis]
    X_valid = X_valid[:, :, np.newaxis]

    print(X_train.shape[0], 'train samples')
    print(X_valid.shape[0], 'validation samples')

    # Convert class vectors to categorical classes matrices
    y_train = np_utils.to_categorical(y_train, parameters.NB_CLASSES)
    y_valid = np_utils.to_categorical(y_valid, parameters.NB_CLASSES)

    print(y_train.shape)
    print(y_valid.shape)

    # Start training
    history = model.fit(X_train, y_train,
                        batch_size=parameters.BATCH_SIZE, epochs=parameters.NB_EPOCH,
                        verbose=parameters.VERBOSE, validation_data=(X_valid, y_valid))

    print('\ntrain finished ::::>>>>    saving model ...')
    model.save(save_dir)


def evaluate_model(model, X_test, y_test):
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')
    X_test = X_test[:, :, np.newaxis]
    print(X_test.shape[0], 'test samples')
    y_test = np_utils.to_categorical(y_test, parameters.NB_CLASSES)
    print(y_test.shape)

    # Start evaluating model with testing data
    score_test = model.evaluate(X_test, y_test, verbose=parameters.VERBOSE)
    print("Testing accuracy:", score_test[1])
