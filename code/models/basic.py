import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Attention, Dense, Dropout, LSTM, Input, Conv1D, Bidirectional, GRU, Flatten, Reshape
from keras import backend as K

# GPU
from tensorflow.compat.v1 import ConfigProto
from keras_self_attention import SeqSelfAttention
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

from models.base_model import ClassificationModel

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
        
        if (self.name.startswith("lstm")):
            inp = Input(shape=(12, 1000))
            x = LSTM(64, activation = 'tanh')(inp)
            out = Dense(num_output_classes, activation='softmax')(x)
            model = Model(inp, out)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:
            pass
        print("[MODEL] Array Sizes")
        print("[MODEL] X_train", X_train.shape)
        print("[MODEL] y_train", y_train.shape)
        print("[MODEL] X_train", X_val.shape)
        print("[MODEL] y_train", y_val.shape)
        history = model.fit(X_train, y_train, batch_size = 256, epochs = 1, validation_data = (X_val, y_val), shuffle = True)
        self.model = model
        np.save(self.outputfolder + "training_history.npy", history.history)
        K.clear_session()
    def predict(self, X):
        X = np.transpose(X, axes=[0, 2, 1])
        return self.model.predict(X)