import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Attention, Dense, Dropout, LSTM, Input, Conv1D, Bidirectional, GRU, Flatten
from keras import backend as K

# GPU
from tensorflow.compat.v1 import ConfigProto
from keras_self_attention import SeqSelfAttention
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

from models.base_model import ClassificationModel

class AttentionModel(ClassificationModel):
    def __init__(self, name, n_classes,  sampling_frequency, outputfolder, input_shape):
        self.name = name
        self.n_classes = n_classes
        self.sampling_frequency = sampling_frequency
        self.outputfolder = outputfolder
        self.input_shape = input_shape
        print("Model", name, ":", self.input_shape, "instantiated.")
    
    def fit(self, X_train, y_train, X_val, y_val):
        if (self.name.startswith("attention_bilstm")):
            inp = Input(shape=(1000,12)) # REVERSE THIS
            x = LSTM(128, activation = 'tanh', return_sequences = True)(inp)
            # x = SeqSelfAttention(attention_activation='tanh')(x)
            out = Dense(71, activation='softmax')(x)
            model = Model(inp, out)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        elif (self.name.startswith("attention_lstm")):
            pass
        elif (self.name.startswith("attention_cnn")):
            pass
        y_train = np.expand_dims(y_train, axis=-1)
        print("[MODEL] Array Sizes")
        print("[MODEL] X_train", X_train.shape)
        print("[MODEL] y_train", y_train.shape)
        history = model.fit(X_train, y_train, batch_size = 64, epochs = 100, validation_data = (X_val, y_val), shuffle = True)
        self.model = model
        np.save(self.outputfolder + "training_history.npy", history.history)
        K.clear_session()
    def predict(self, X):
        return self.model.predict(X)