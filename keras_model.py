#
# The SELDnet architecture
#

from tensorflow.keras.layers import Bidirectional, Conv2D, MaxPooling2D, Input, Concatenate
from tensorflow.keras.layers import Dense, Activation, Dropout, Reshape, Permute
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow import keras
import tensorflow as tf
# keras.backend.set_image_data_format('channels_first')
from IPython import embed
import numpy as np


def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, f_pool_size, t_pool_size,
              rnn_size, fnn_size, weights, doa_objective, is_accdoa):
    # model definition
    spec_start = Input(shape=(data_in[-2], data_in[-1], data_in[-3]))

    # CNN
    spec_cnn = spec_start
    for i, convCnt in enumerate(f_pool_size):
        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(t_pool_size[i], f_pool_size[i]))(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)
    spec_cnn = Permute((1, 3, 2))(spec_cnn)

    # RNN    
    spec_rnn = Reshape((data_out[-2] if is_accdoa else data_out[0][-2], -1))(spec_cnn)
    for nb_rnn_filt in rnn_size:
        spec_rnn = Bidirectional(
            GRU(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                return_sequences=True),
            merge_mode='mul'
        )(spec_rnn)

    # FC - DOA
    doa = spec_rnn
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filt))(doa)
        doa = Dropout(dropout_rate)(doa)

    doa = TimeDistributed(Dense(data_out[-1] if is_accdoa else data_out[1][-1]))(doa)
    doa = Activation('tanh', name='doa_out')(doa)

    model = None
    if is_accdoa:
        model = Model(inputs=spec_start, outputs=doa)
        model.compile(optimizer=Adam(), loss='mse')
    else:
        # FC - SED
        sed = spec_rnn
        for nb_fnn_filt in fnn_size:
            sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
            sed = Dropout(dropout_rate)(sed)
        sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
        sed = Activation('sigmoid', name='sed_out')(sed)

        if doa_objective is 'mse':
            model = Model(inputs=spec_start, outputs=[sed, doa])
            model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=weights)
        elif doa_objective is 'masked_mse':
            doa_concat = Concatenate(axis=-1, name='doa_concat')([sed, doa])
            model = Model(inputs=spec_start, outputs=[sed, doa_concat])
            model.compile(optimizer=Adam(), loss=['binary_crossentropy', masked_mse], loss_weights=weights)
        else:
            print('ERROR: Unknown doa_objective: {}'.format(doa_objective))
            exit()
    model.summary()
    return model


def masked_mse(y_gt, model_out):
    nb_classes = 12 #TODO fix this hardcoded value of number of classes
    # SED mask: Use only the predicted DOAs when gt SED > 0.5
    sed_out = y_gt[:, :, :nb_classes] >= 0.5 
    sed_out = keras.backend.repeat_elements(sed_out, 3, -1)
    sed_out = keras.backend.cast(sed_out, 'float32')

    # Use the mask to computed mse now. Normalize with the mask weights 
    return keras.backend.sqrt(keras.backend.sum(keras.backend.square(y_gt[:, :, nb_classes:] - model_out[:, :, nb_classes:]) * sed_out))/keras.backend.sum(sed_out)


def load_seld_model(model_file, doa_objective):
    if doa_objective is 'mse':
        return load_model(model_file)
    elif doa_objective is 'masked_mse':
        return load_model(model_file, custom_objects={'masked_mse': masked_mse})
    else:
        print('ERROR: Unknown doa objective: {}'.format(doa_objective))
        exit()



