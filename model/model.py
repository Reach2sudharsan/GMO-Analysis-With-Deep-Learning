from email import generator
from keras.layers import Conv1D, Conv1DTranspose, BatchNormalization,LSTM, Dense, Embedding
from keras.layers import Input, Dropout, Bidirectional, GlobalMaxPooling1D, Activation, Concatenate
from keras.initializers import RandomNormal
from keras.models import Sequential, Model
from keras.optimizers import adam_v2
from keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings(action = 'ignore')

# generator a resnet block
def resnet_block(n_filters, input_layer):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# first layer convolutional layer
	g = Conv1D(n_filters, 3, padding='same', kernel_initializer=init)(input_layer)
	g = BatchNormalization(axis=-1)(g)
	g = Activation('sigmoid')(g)
	# second convolutional layer
	g = Conv1D(n_filters, 3, padding='same', kernel_initializer=init)(g)
	g = BatchNormalization(axis=-1)(g)
	# concatenate merge channel-wise with input layer
	g = Concatenate()([g, input_layer])
	return g

# define the generator model
def build_generator(spec_shape, n_resnet=9):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_spec = Input(shape=spec_shape)
    g = Embedding(spec_shape,128,input_length=120)(in_spec)
    g = Conv1D(64, 7, padding='same', kernel_initializer=init)(g)
    g = BatchNormalization(axis=-1)(g)
    g = Activation('sigmoid')(g)
    # d128
    g = Conv1D(128, 3, strides=2, padding='same', kernel_initializer=init)(g)
    g = BatchNormalization(axis=-1)(g)
    g = Activation('sigmoid')(g)
    # d256
    g = Conv1D(256, 3, strides=2, padding='same', kernel_initializer=init)(g)
    g = BatchNormalization(axis=-1)(g)
    g = Activation('sigmoid')(g)
    # R256
    for _ in range(n_resnet):
        g = resnet_block(256, g)
    # u128
    g = Conv1DTranspose(128, 3, strides=2, padding='same', kernel_initializer=init)(g)
    g = BatchNormalization(axis=-1)(g)
    g = Activation('sigmoid')(g)
    # u64
    g = Conv1DTranspose(64, 3, strides=2, padding='same', kernel_initializer=init)(g)
    g = BatchNormalization(axis=-1)(g)
    g = Activation('sigmoid')(g)

    g = Conv1D(44, 7, padding='same', kernel_initializer=init)(g)
    g = BatchNormalization(axis=-1)(g)
    out_spec = Activation('sigmoid')(g)
    # define model
    generator = Model(in_spec, out_spec)
    generator.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
    return generator

def build_BiLSTM(size_of_vocabulary):
    model = Sequential()
    #embedding layer
    model.add(Embedding(size_of_vocabulary,128,input_length=10))
    #lstm layer
    model.add(Bidirectional(LSTM(30,return_sequences=True,dropout=0.2)))
    model.add(Bidirectional(LSTM(60,return_sequences=True,dropout=0.2)))
    #Global Maxpooling
    model.add(GlobalMaxPooling1D())
    #Dense Layer
    model.add(Dense(60,activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(120,activation='sigmoid'))
    #Add loss function, metrics, optimizer
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    #Adding callbacks
    es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=3)
    mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True,verbose=1)
    #summary
    model.summary()
    return model, es, mc

gen = build_BiLSTM(40000)