import imp
import streamlit as st
from sklearn.model_selection import train_test_split
import sklearn.datasets as sk
import pandas as pd
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from model import model_making,encoded_data
from autokerasmodel import autokeras_classification,autokeras_regression

def methodofkeys(dict, search_iter):
    for iter1, iter2 in dict.items():
        if iter2 == search_iter:
            return iter1

list_of_optimizers=['SGD','Adadelta','Adam','Adagrad','Adamax','Ftrl','Nadam','RMSprop']
list_of_loss_func=dict({'mse': ' mse : Computes the mean squared error between labels and predictions.',
'binary_crossentropy': ' binary_crossentropy : Computes the binary crossentropy loss.',
'categorical_crossentropy': 'categorical_crossentropy : Computes the categorical crossentropy loss.',
'categorical_hinge': ' categorical_hinge : Computes the categorical hinge loss between y_true and y_pred.',
'cosine_similarity': 'cosine_similarity : Computes the cosine similarity between labels and predictions.',
'hinge': ' hinge : Computes the hinge loss between y_true and y_pred.',
'huber': ' huber : Computes Huber loss value.',
'kld': ' kld : Computes Kullback-Leibler divergence loss between y_true and y_pred.',
'log_cosh': ' log_cosh : Logarithm of the hyperbolic cosine of the prediction error.',
'mae': ' mae : Computes the mean absolute error between labels and predictions.',
'mape': ' mape : Computes the mean absolute percentage error between y_true and y_pred.',
'msle': ' msle : Computes the mean squared logarithmic error between y_true and y_pred.',
'poisson': ' poisson : Computes the Poisson loss between y_true and y_pred.',
'sparse_categorical_crossentropy': ' sparse_categorical_crossentropy : Computes the sparse categorical crossentropy loss.',
'squared_hinge': ' squared_hinge : Computes the squared hinge loss between y_true and y_pred'})
list_of_metrics_func=list_of_loss_func.copy()
list_of_metrics_func['top_k_categorical_accuracy']='top_k_categorical_accuracy : Computes how often targets are in the top K predictions.'
list_of_metrics_func['sparse_categorical_accuracy']='sparse_categorical_accuracy : Calculates how often predictions equal labels.'
list_of_metrics_func['sparse_top_k_categorical_accuracy']='sparse_top_k_categorical_accuracy :  Computes how often integer targets are in the top K predictions.'
list_of_metrics_func['categorical_accuracy']='categorical_accuracy(...): Calculates how often predictions match one-hot labels.'
list_of_metrics_func['accuracy']='accuracy(...): Calculates how often predictions equal labels.'
st.set_page_config(page_title='Neural Network Playground', page_icon='🏛️')
st.title("Neural Network Playground")
with st.expander('Dataset'): 
    is_dataset_uploaded_or_dataset_from_libraries=st.radio("Upload Dataset or Preconceived Dataset",['Upload Dataset','Preconceived Dataset'])
    if(is_dataset_uploaded_or_dataset_from_libraries=='Upload Dataset'):
        data_train_file=st.file_uploader("Upload Training Dataset")
        is_val_file_exist=st.selectbox('Is there a validation dataset',['No','Yes'])
        if(is_val_file_exist=='Yes'):
            data_val_file=st.file_uploader("Upload Validation Dataset")
        is_test_file_exist=st.selectbox('Is there a testing dataset',['No','Yes'])
        if(is_test_file_exist=='Yes'):
            data_test_file=st.file_uploader("Upload Testing Dataset")
        if(data_train_file is not None and is_val_file_exist=='No' and is_test_file_exist=='No'):
            train_data=pd.read_csv(data_train_file)
            columns_for_data=st.multiselect('Which columns are required for training and testing',train_data.columns)
            columns_for_values=st.multiselect('Which columns are output columns',train_data.columns)
            actual_data=train_data[columns_for_data]
            actual_values=train_data[columns_for_values]
            data_train_val,data_test,value_train_val,value_test=train_test_split(actual_data,actual_values,train_size=0.7,random_state=42)
            data_train,data_val,value_train,value_val=train_test_split(data_train_val,value_train_val,train_size=0.7,random_state=42)
        elif(data_train_file is not None and is_val_file_exist=='Yes' and is_test_file_exist=='No'):
            train_data=pd.read_csv(data_train_file)
            val_data=pd.read_csv(data_val_file)
            columns_for_data=st.multiselect('Which columns are required for training and testing',train_data.columns)
            columns_for_values=st.multiselect('Which columns are output columns',train_data.columns)
            actual_data=train_data[columns_for_data]
            data_val=val_data[columns_for_data]
            actual_values=train_data[columns_for_values]
            value_val=val_data[columns_for_values]
            data_train,data_test,value_train,value_test=train_test_split(actual_data,actual_values,train_size=0.7,random_state=42)
        elif(data_train_file is not None and is_val_file_exist=='No' and is_test_file_exist=='Yes'):
            train_data=pd.read_csv(data_train_file)
            test_data=pd.read_csv(data_test_file)
            columns_for_data=st.multiselect('Which columns are required for training and testing',train_data.columns)
            columns_for_values=st.multiselect('Which columns are output columns',train_data.columns)
            actual_data=train_data[columns_for_data]
            data_test=test_data[columns_for_data]
            actual_values=train_data[columns_for_values]
            value_test=test_data[columns_for_values]
            data_train,data_val,value_train,value_val=train_test_split(actual_data,actual_values,train_size=0.7,random_state=42)
        elif(data_train_file is not None and is_val_file_exist=='Yes' and is_test_file_exist=='Yes'):
            train_data=pd.read_csv(data_train_file)
            test_data=pd.read_csv(data_test_file)
            val_data=pd.read_csv(data_val_file)
            columns_for_data=st.multiselect('Which columns are required for training and testing',train_data.columns)
            columns_for_values=st.multiselect('Which columns are output columns',train_data.columns)
            data_train=train_data[columns_for_data]
            data_val=val_data[columns_for_data]
            data_test=test_data[columns_for_data]
            value_train=train_data[columns_for_values]
            value_test=test_data[columns_for_values]
            value_val=val_data[columns_for_values]
        elif(data_train_file is not None):
            st.error('No dataset uploaded')
        

    if(is_dataset_uploaded_or_dataset_from_libraries=='Preconceived Dataset'):
        datasets=dict({'None':None,'California Housing Dataset':sk.fetch_california_housing(as_frame=True),'Iris dataset':sk.load_iris(as_frame=True),'Diabetes datset':sk.load_diabetes(as_frame=True),'Wine Dataset':sk.load_wine(as_frame=True),'Linnerud Datset':sk.load_linnerud(as_frame=True)})
        option=st.selectbox("Choose the preconceived datatset",datasets.keys())
        columns_for_data=st.multiselect('Which columns are required for training and testing',datasets[option].frame.columns)
        columns_for_values=st.multiselect('Which columns are output columns',datasets[option].frame.columns)
        if option=='None':
           st.error('No dataset detected')
        elif  option!='None':
           actual_data=datasets[option].frame[columns_for_data]
           actual_values=datasets[option].frame[columns_for_values]
        data_train_val,data_test,value_train_val,value_test=train_test_split(actual_data,actual_values,train_size=0.7,random_state=42)
        data_train,data_val,value_train,value_val=train_test_split(data_train_val,value_train_val,train_size=0.7,random_state=42)
    c_or_r=dict({'Classification':'c','Regression':'r'})
    c_r=st.radio('Type of dataset: ',c_or_r.keys())
    c_r=c_or_r[c_r]
    button_1=st.button("Basic information about data")
    if(button_1):
        st.write('Type of Dataset = '+methodofkeys(c_or_r,c_r))
        st.write('Length of Data = ', len(actual_data))
        st.write('No. of features = ', len(columns_for_data))
        if(c_r=='c'):
            st.write('No. of classes = ', len(actual_values.value_counts()))
        elif(c_r=='r'):
            st.write('No. of target variables = ',len(columns_for_values))  
        st.dataframe(actual_data.describe())
    

type_of_method=['None','AutoML','Custom']
method_type=st.radio('Type of Process Of Finding Model',type_of_method)
if(method_type=='AutoML'):
        num_of_trial=st.number_input("No. of model to be trialed",min_value=2)
        new_Loss_func=methodofkeys(list_of_loss_func,st.selectbox('Loss Function',list_of_loss_func.values()))
        new_list_of_metrics_function=st.multiselect('Metrics for the model',list_of_metrics_func.values())
        new_keys_of_metric_function=[methodofkeys(list_of_metrics_func,i) for i in new_list_of_metrics_function]
        new_epoch=st.number_input("No. of Epoch",min_value=1)
        if st.button('Find the best model'):
            if(c_r=='c'):
                predicted_values,metrics_value=autokeras_classification(data_train,value_train,data_val,value_val,data_test,value_test,num_of_trial,new_epoch,new_Loss_func,new_keys_of_metric_function)
                metrics_value=pd.DataFrame(metrics_value)
                st.dataframe(metrics_value)
            
            if(c_r=='r'):
                predicted_values,metrics_value,best_model=autokeras_regression(data_train,value_train,data_val,value_val,data_test,value_test,num_of_trial,new_epoch,new_Loss_func,new_keys_of_metric_function)
                metrics_value=pd.DataFrame(metrics_value)
                new_columns=[]
                new_columns.append(new_Loss_func)
                for i in new_keys_of_metric_function:
                    new_columns.append(i)
                metrics_value['Type of Metrics']=new_columns
                c = metrics_value.columns
                metrics_value = metrics_value[c[np.r_[1, 0, 2:len(c)]]]
                st.dataframe(best_model.summary())
                st.dataframe(metrics_value)

if(method_type==('Custom')):
    nooflayers=st.number_input("No.of hidden layers",min_value=1)
    length=[]
    list_of_af=['elu','exponential','gelu','hard_sigmoid','linear','relu','selu','sigmoid','softmax','softplus','softsign','swish','tanh']
    activation_function=[]
    for i in range(nooflayers):
        with st.expander('Layer '+ str(i+1)):
            length.append(st.number_input("No of nodes of layer " +str(i+1),min_value=1))
            activation_function.append(st.selectbox('Activation Function of layer '+str(i+1),list_of_af))
    if(c_r=='c'):
        activation_function.append(st.selectbox('Activation Function of output layer',list_of_af))
    Optimizers=st.selectbox('Optimizers',list_of_optimizers)
    Loss_func=methodofkeys(list_of_loss_func,st.selectbox('Loss Function',list_of_loss_func.values()))
    list_of_metrics_function=st.multiselect('Metrics for the model',list_of_metrics_func.values())
    keys_of_metric_function=[methodofkeys(list_of_metrics_func,i) for i in list_of_metrics_function]
    epoch=st.number_input("No. of Epoch",min_value=1)

    if st.button('Run the given model'):
        model=model_making(data_train,value_train,nooflayers,length,activation_function,c_r,Optimizers,Loss_func,keys_of_metric_function,epoch)
        fit=model.fit(data_train,encoded_data(value_train),epochs=epoch,validation_split=0.3)
        history = pd.DataFrame.from_dict(fit.history)
        st.dataframe(history)
        val=model.evaluate(data_val,encoded_data(value_val),verbose=1)
        st.write(val)




    






    





