import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Attention, Dense, Dropout, LSTM, Input, Conv1D, Bidirectional, GRU, Flatten, Reshape
from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import Adam

# GPU
from tensorflow.compat.v1 import ConfigProto
from keras_self_attention import SeqSelfAttention
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

from models.base_model import ClassificationModel

def stacked_lstm(units=[128, 128], lstm_activation = 'tanh', num_output_classes = 71, add_dense = False):
    inp = Input(shape=(12, 1000))
    
    if len(units) == 1:
        x = LSTM(units[0], activation = lstm_activation, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01))
    else:
        x = LSTM(units[0], activation = lstm_activation, return_sequences = True, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01))

    for i in range(1, len(units)):
        if i != len(units) - 1:
            x = LSTM(units[i], activation = lstm_activation, return_sequences = True, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01))(inp)
        else:
            x = LSTM(units[i], activation = lstm_activation, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01))(inp)

    if units[-1] > 128 and add_dense:
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)

    if add_dense and num_output_classes < 64:
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)

    out = Dense(num_output_classes, activation='softmax')(x)
    model = Model(inp, out)
    return model

class BasicModel(ClassificationModel):
    def __init__(self, name, n_classes,  sampling_frequency, outputfolder, input_shape):
        self.name = name
        self.n_classes = n_classes
        self.sampling_frequency = sampling_frequency
        self.outputfolder = outputfolder
        self.input_shape = input_shape
        print("Model", name, ":", self.input_shape, "instantiated.")
    
    def fit(self, X_train, y_train, X_val, y_val):
        X_train = np.transpose(X_train, axes=[0, 2, 1])
        X_val = np.transpose(X_val, axes=[0, 2, 1])
        num_output_classes = y_train.shape[1]
        
        if (self.name.startswith("stacked_lstm_256_256")):
            model = stacked_lstm(units = [256, 256], num_output_classes = num_output_classes, add_dense = True)
        
        elif (self.name.startswith("stacked_lstm_128_128")):
            model = stacked_lstm(units = [128, 128], num_output_classes = num_output_classes, add_dense = True)
        
        elif (self.name.startswith("stacked_lstm_64_64")):
            model = stacked_lstm(units = [64, 64], num_output_classes = num_output_classes, add_dense = False)
        
        elif (self.name.startswith("stacked_lstm_256_128")):
            model = stacked_lstm(units = [256, 128], num_output_classes = num_output_classes, add_dense = True)
        
        elif (self.name.startswith("stacked_lstm_128_64")):
            model = stacked_lstm(units = [128, 64], num_output_classes = num_output_classes, add_dense = False)
        else:
        	print("Model", self.name, "not found")
        	exit(0)
        adam = Adam(lr=1e-4)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        print("[MODEL] Array Sizes")
        print("[MODEL] X_train", X_train.shape)
        print("[MODEL] y_train", y_train.shape)
        print("[MODEL] X_train", X_val.shape)
        print("[MODEL] y_train", y_val.shape)
        history = model.fit(X_train, y_train, batch_size = 256, epochs = 100, validation_data = (X_val, y_val), shuffle = True)
        self.model = model
        np.save(self.outputfolder + "training_history.npy", history.history)
        K.clear_session()
    def predict(self, X):
        X = np.transpose(X, axes=[0, 2, 1])
        return self.model.predict(X)