from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.layers import Convolution1D, AveragePooling1D, LSTM
from keras.callbacks import EarlyStopping


def cnn_2d(shape=(16, 600, 1)):
    model = Sequential()
    model.add(Convolution2D(64, 3, 1, border_mode="same",
              activation="relu", input_shape=shape))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    return model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


def cnn_1d(shape=(9600, 1)):
    model = Sequential()
    model.add(Convolution1D(nb_filter=6, filter_length=3,
              border_mode="valid", activation="relu", input_shape=shape))
    model.add(AveragePooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    return model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


def lstm_1(shape=(9600, 1)):
    model = Sequential()
    model.add(LSTM(4, input_shape=shape)
    model.add(Dense(1))
    return model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

def cnn_artrous():
    cnn = Sequential()
    cnn.add(AtrousConvolution2D(64, 3, 1, atrous_rate=(2,2), border_mode='valid', input_shape=(16, 600, 1)))
    cnn.add(Activation('relu'))
    cnn.add(PReLU())
    cnn.add(AveragePooling2D(pool_size=(2, 1)))
    cnn.add(AtrousConvolution2D(64, 3, 1, atrous_rate=(2,2), border_mode='valid'))
    cnn.add(Activation('relu'))
    cnn.add(PReLU())
    cnn.add(AveragePooling2D(pool_size=(2, 1)))
    cnn.add(Flatten())
    cnn.add(Dense(512, init='he_normal'))
    cnn.add(PReLU())
    cnn.add(Dropout(0.5))
    cnn.add(Dense(512, init='he_normal'))
    cnn.add(PReLU())
    cnn.add(Dropout(0.5))
    cnn.add(Dense(1, init='he_normal', activation='sigmoid'))
    return cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
