import pandas as pd
import numpy as np
import re

def clean(x):
    return x.replace('[0][0]','').replace('(','').replace(')','').replace('[','').replace(']','')

def magic(x):
    tmp = re.split(r'\s{1,}',clean(x))
    return tmp

def normalize(df): 
    for column in df.columns:
        df[column] = df[column]  / df[column].abs().max()
    
    return df
def standization(df):
    for column in df.columns:
        df[column]=(df[column] - np.average(df[column])) / (np.std(df[column]))
    
    return df

def get_model_summary(model):
    width = 250
    stringlist = []
    model.summary(width, print_fn=lambda x: stringlist.append(x))
    summ_string = "".join(stringlist)
    splitstr1 = f"={{{width}}}"
    splitstr2 = f"_{{{width}}}"
    tmptable = re.split(splitstr1,summ_string)
    header = re.split(splitstr2, tmptable[0])
    header = re.split(r'\s{2,}', header[1])[:-1]
    table = re.split(splitstr2, tmptable[1])
    
    df = pd.DataFrame(columns=header)
    for index,entry in enumerate(table):
        entry = re.split(r'\s{2,}', entry)[:-1]
        df.loc[index] = {header[0] : entry[0],
                     header[1] : tuple([int(e) for e in clean(entry[1]).split(', ')[1:]]),
                     header[2] : int(entry[2])}
    
    return df

def detect_and_removing_outliers(dataset):
    for i in dataset.columns:
        Q1 = np.percentile(dataset[i], 25,
                   interpolation = 'midpoint')
        Q3 = np.percentile(dataset[i], 75,
                   interpolation = 'midpoint')
        IQR = Q3 - Q1
        upper = np.where(dataset[i] >= (Q3+1.5*IQR))
        lower = np.where(dataset[i] <= (Q1-1.5*IQR))
        dataset.drop(upper[0], inplace = True)
        dataset.drop(lower[0], inplace = True)
    new_shape=(dataset.shape)
    return new_shape,dataset

def methodofkeys(dict, search_iter):
    for iter1, iter2 in dict.items():
        if iter2 == search_iter:
            return iter1