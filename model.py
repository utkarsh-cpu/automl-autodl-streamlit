import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.datasets as sk
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras

def encoded_data(data_values):
    onehot_encoded=keras.utils.to_categorical(data_values)
    return onehot_encoded


def model_making(train_data,train_value,nooflayers,length,activation_function,c_r,optimizers,loss_function,metrics,epoch):
    model=keras.Sequential()
    model.add(tf.keras.Input(shape=(len(train_data[0]),)))
    for i in range(nooflayers):
        model.add(tf.keras.layers.Dense(length[i],activation=activation_function[i]))
    if(c_r=='c'):
        model.add(tf.keras.layers.Dense(len(pd.DataFrame(train_value).value_counts()),activation=activation_function[-1]))
        model.compile(optimizer=optimizers,loss=loss_function,metrics=metrics)
    elif(c_r=='r'):
        model.add(tf.keras.layers.Dense(len(train_value[0])))
        model.compile(optimizer=optimizers,loss=loss_function,metrics=metrics)
    return model
    

