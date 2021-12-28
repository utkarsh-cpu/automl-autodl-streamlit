import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.datasets as sk
from tensorflow import keras


def model_making(train_data,train_value,nooflayers,length,activation_function,c_r,optimizers,loss_function,metrics,epoch):
    model=keras.Sequential()
    model.add(tf.keras.Input(size=(len(train_data[0]),)))
    for i in range(nooflayers):
        model.add(tf.keras.Dense(length[i],activation=activation_function[i]))
    if(c_r=='c'):
        model.add(tf.keras.Dense(len(train_value.value_counts())))
    elif(c_r=='r'):
        model.add(tf.keras.Dense(len(train_value[0])))
    
    model.compile(optimizer=optimizers,loss=loss_function,metrics=metrics)
    model.fit(train_data,train_value,epoch)
    

