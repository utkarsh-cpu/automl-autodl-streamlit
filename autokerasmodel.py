import imp
import autokeras as ak
from tensorflow import keras

def encoded_data(data_values):
    onehot_encoded=keras.utils.to_categorical(data_values)
    return onehot_encoded

def autokeras_classification(data,values,is_val_data_present,no_of_model_to_be_checked,epoch,loss_func,metrics,data_val=None,value_val=None):
    model=ak.StructuredDataClassifier(max_trials=no_of_model_to_be_checked,loss=loss_func,metrics=metrics)
    if(is_val_data_present=='Yes'):
        fit=model.fit(data,encoded_data(values),validation_data=(data_val,encoded_data(value_val)),epochs=epoch)
    elif(is_val_data_present=='No'):
        fit=model.fit(data,encoded_data(values),epochs=epoch,validation_split=0.3)
    '''
    predict_y=model.predict(test_data)
    metrics_value=(model.evaluate(test_data,encoded_data(test_values)))
    '''
    model_best = model.export_model()
    return model,fit,model_best

def autokeras_regression(data,values,is_val_data_present,no_of_model_to_be_checked,epoch,loss_func,metrics,data_val=None,value_val=None):
    model=ak.StructuredDataRegressor(max_trials=no_of_model_to_be_checked,loss=loss_func,metrics=metrics)
    if(is_val_data_present=='Yes'):
       fit=model.fit(data,(values),validation_data=(data_val,encoded_data(value_val)),epochs=epoch)
    elif(is_val_data_present=='No'):
        fit=model.fit(data,(values),epochs=epoch,validation_split=0.3)
    '''
    predict_y=model.predict(test_data)
    metrics_value=(model.evaluate(test_data,test_values))
    '''
    model_best = model.export_model()
    return model,fit,model_best



