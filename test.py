import streamlit as st
import pandas as pd
import sklearn.datasets as sk
import time 
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from model import model_making,encoded_data
from autokerasmodel import autokeras_classification,autokeras_regression
from imp_function import detect_and_removing_outliers, normalize,standization,methodofkeys,get_model_summary

st.set_page_config(page_title='Neural Network Playground', page_icon='🏛️')

Content=['Uploading the Datasets',
        'Preprocessing of Data',
        'Basic Information About Your Data',
        'Type of Process for Neural Network Search', 
        'Training and Finding Model',
        'Testing Model on Testing Data']

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


st.sidebar.header("Neural Network Playground")

st.session_state.position=st.sidebar.radio("Content",Content,key='2')


if(st.session_state.position=='Uploading the Datasets'):
    st.header('Upload Datasets or Use Preconceived Dataset')
    pre_dataset_or_upload_dataset=st.radio("Type",["Use Uploaded Dataset","Use Preconceived Dataset"])
    if(pre_dataset_or_upload_dataset=='Use Uploaded Dataset'):
        type_of_file=st.selectbox("Type of File",['CSV','Excel'])
        train_file=st.file_uploader("Upload Training Dataset")
        if(st.button("Upload Training Dataset")):
            if(type_of_file=='CSV'):
                st.session_state.train_data=pd.read_csv(train_file)    
            else:
                st.session_state.train_data=pd.read_excel(train_file)
            st.success('Dataset Uploaded, You Can Go to Next if You Do Not have Validation Dataset')
        
        st.session_state.is_val_data=st.radio("Is there a validation dataset",['No','Yes'])
        if(st.session_state.is_val_data=='No'):
            st.session_state.val_data=None
        if(st.session_state.is_val_data=='Yes'):
            val_file=st.file_uploader("Upload Validation Dataset")
            if(st.button('Upload Val Dataset')):
                if(type_of_file=='CSV'):
                    st.session_state.val_data=pd.read_csv(val_file)
                else:
                    st.session_state.val_data=pd.read_excel(val_file)
                st.success('Dataset Uploaded, You Can Go to Next')
    
    if(pre_dataset_or_upload_dataset=='Use Preconceived Dataset'):
        datasets=dict({'None':None,'California Housing Dataset':sk.fetch_california_housing(as_frame=True),'Iris dataset':sk.load_iris(as_frame=True),'Diabetes datset':sk.load_diabetes(as_frame=True),'Wine Dataset':sk.load_wine(as_frame=True),'Linnerud Datset':sk.load_linnerud(as_frame=True)})
        option=st.selectbox("Choose the preconceived datatset",datasets.keys())
        if(st.button("Upload")):
            st.session_state.train_data=datasets[option].frame
            st.success('Dataset Uploaded, You Can Go to Next')


if(st.session_state.position=='Preprocessing of Data'):
    st.session_state.c_or_r=dict({'Classification':'c','Regression':'r'})
    c_r=st.radio('Type of dataset: ',st.session_state.c_or_r.keys())
    st.session_state.c_r=st.session_state.c_or_r[c_r]
    st.session_state.data_columns=st.multiselect("Which Columns are required for data",st.session_state.train_data.columns)
    st.session_state.value_columns=st.multiselect("Which Columns are required for value",st.session_state.train_data.columns)
    st.session_state.index_columns=st.multiselect("Which Columns are required for indexing(For Date Columns)",st.session_state.train_data.columns)
    if(st.session_state.is_val_data=='Yes'):
        if(st.session_state.train_data.isnull().values.any()):
           null_value=st.selectbox("This dataset has null values, which way do you want to deal with it",['Removing the null values','Filling NA values with mean'])
           if(null_value=='Removing the null values'):
                st.session_state.train_data.dropna(inplace=True)
                st.session_state.val_data.dropna(inplace=True)
           else:
                st.session_state.train_data.fillna(st.session_state.train_data.mean(),inplace=True)
                st.session_state.val_data.fillna(st.session_state.val_data.mean(),inplace=True)
        with(st.spinner("Detecting and Removing Outliers..")):
            new_shape,st.session_state.train_data=detect_and_removing_outliers(st.session_state.train_data)
            new_val_shape,st.session_state.val_data=detect_and_removing_outliers(st.session_state.val_data)
            time.sleep(2)
        st.success('Successfully Removed Outliers with new shape'+str(new_shape))        
        is_data_to_be_normalized=st.selectbox("Is Data to be Normalized",['No','Yes'])
        if(is_data_to_be_normalized=='Yes'):
            st.session_state.train_data=normalize(st.session_state.train_data)
            st.session_state.val_data=normalize(st.session_state.val_data)
        is_data_to_be_standardized=st.selectbox("Is Data to be Standardized",['No','Yes'])
        if(is_data_to_be_standardized=='Yes'):
            st.session_state.train_data=standization(st.session_state.train_data)
            st.session_state.val_data=standization(st.session_state.val_data)
        st.session_state.data_train=st.session_state.train_data[st.session_state.data_columns]
        st.session_state.value_train=st.session_state.train_data[st.session_state.value_columns]
        st.session_state.data_val=st.session_state.val_data[st.session_state.data_columns]
        st.session_state.value_val=st.session_state.val_data[st.session_state.value_columns]
    elif(st.session_state.is_val_data=='No'):
        if(st.session_state.train_data.isnull().values.any()):
           null_value=st.selectbox("This dataset has null values, which way do you want to deal with it",['Removing the null values','Filling NA values with mean'])
           if(null_value=='Removing the null values'):
                st.session_state.train_data.dropna(inplace=True)
           else:
                st.session_state.train_data.fillna(st.session_state.train_data.mean(),inplace=True)
        with(st.spinner("Detecting and Removing Outliers..")):
            new_shape,st.session_state.train_data=detect_and_removing_outliers(st.session_state.train_data)
            time.sleep(2)
        st.success('Successfully Removed Outliers with new shape'+str(new_shape))        
        is_data_to_be_normalized=st.selectbox("Is Data to be Normalized",['No','Yes'])
        if(is_data_to_be_normalized=='Yes'):
            st.session_state.train_data=normalize(st.session_state.train_data)
        is_data_to_be_standardized=st.selectbox("Is Data to be Standardized",['No','Yes'])
        if(is_data_to_be_standardized=='Yes'):
            st.session_state.train_data=standization(st.session_state.train_data)
        st.session_state.data_train=st.session_state.train_data[st.session_state.data_columns]
        st.session_state.value_train=st.session_state.train_data[st.session_state.value_columns]
        st.session_state.data_val=None
        st.session_state.value_val=None
    if(st.button("Next")):
        st.success('Yes you can go to next section')

if(st.session_state.position=='Basic Information About Your Data'):
    st.header("Basic Information About The Data")
    st.subheader('Type of Dataset = '+methodofkeys(st.session_state.c_or_r,st.session_state.c_r))
    st.subheader('Length of Data = '+str(len(st.session_state.train_data)))
    st.subheader('No. of features = '+str(len(st.session_state.data_columns)))
    if(st.session_state.c_r=='c'):
        st.subheader('No. of classes = '+str(len(st.session_state.train_data[st.session_state.value_columns].value_counts())))
    elif(st.session_state.c_r=='r'):
        st.subheader('No. of target variables = '+str(len(st.session_state.train_data[st.session_state.value_columns]))) 
    st.dataframe(st.session_state.train_data[st.session_state.data_columns].describe())

if(st.session_state.position=='Type of Process for Neural Network Search'):
    type_of_method=['AutoML','Custom']
    st.session_state.method_type=st.radio('Type of Process Of Finding Model',type_of_method)
    if(st.button("Submit")):
        st.success("Successfully Submitted")

if(st.session_state.position=='Training and Finding Model'):
    if(st.session_state.method_type=='AutoML'):
        num_of_trial=st.number_input("No. of model to be trialed",min_value=2)
        new_Loss_func=methodofkeys(list_of_loss_func,st.selectbox('Loss Function',list_of_loss_func.values()))
        new_list_of_metrics_function=st.multiselect('Metrics for the model',list_of_metrics_func.values())
        new_keys_of_metric_function=[methodofkeys(list_of_metrics_func,i) for i in new_list_of_metrics_function]
        new_epoch=st.number_input("No. of Epoch",min_value=1)
        if st.button('Find the best model'):
            if(st.session_state.c_r=='c'):
                model,fit,best_model=autokeras_classification(st.session_state.data_train,st.session_state.value_train,st.session_state.is_val_data,num_of_trial,new_epoch,new_Loss_func,new_keys_of_metric_function,st.session_state.data_val,st.session_state.value_val)
                history = pd.DataFrame.from_dict(fit.history)
                st.dataframe(history)
                st.dataframe(get_model_summary(best_model))
                
            
            if(st.session_state.c_r=='r'):
                model,fit,best_model=autokeras_regression(st.session_state.data_train,st.session_state.value_train,st.session_state.is_val_data,num_of_trial,new_epoch,new_Loss_func,new_keys_of_metric_function,st.session_state.data_val,st.session_state.value_val)
                history = pd.DataFrame.from_dict(fit.history)
                st.dataframe(history)
                st.dataframe(get_model_summary(best_model))

    if(st.session_state.method_type==('Custom')):
        nooflayers=st.number_input("No.of hidden layers",min_value=1)
        length=[]
        list_of_af=['elu','exponential','gelu','hard_sigmoid','linear','relu','selu','sigmoid','softmax','softplus','softsign','swish','tanh']
        activation_function=[]
        for i in range(nooflayers):
            with st.expander('Layer '+ str(i+1)):
                length.append(st.number_input("No of nodes of layer " +str(i+1),min_value=1))
                activation_function.append(st.selectbox('Activation Function of layer '+str(i+1),list_of_af))
        if(st.session_state.c_r=='c'):
            activation_function.append(st.selectbox('Activation Function of output layer',list_of_af))
        Optimizers=st.selectbox('Optimizers',list_of_optimizers)
        Loss_func=methodofkeys(list_of_loss_func,st.selectbox('Loss Function',list_of_loss_func.values()))
        list_of_metrics_function=st.multiselect('Metrics for the model',list_of_metrics_func.values())
        keys_of_metric_function=[methodofkeys(list_of_metrics_func,i) for i in list_of_metrics_function]
        epoch=st.number_input("No. of Epoch",min_value=1)

        if st.button('Run the given model'):
            model=model_making(st.session_state.data_train,st.session_state.value_train,nooflayers,length,activation_function,c_r,Optimizers,Loss_func,keys_of_metric_function,epoch)
            fit=model.fit(st.session_state.data_train,encoded_data(st.session_state.value_train),epochs=epoch,validation_split=0.3)
            history = pd.DataFrame.from_dict(fit.history)
            st.dataframe(history)
    
    
        
        



                
            
