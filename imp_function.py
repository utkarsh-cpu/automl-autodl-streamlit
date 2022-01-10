import pandas as pd
import numpy as np

def normalize(df): 
    for column in df.columns:
        df[column] = df[column]  / df[column].abs().max()
    
    return df
def standization(df):
    for column in df.columns:
        df[column]=(df[column] - np.average(df[column])) / (np.std(df[column]))
    
    return df
    