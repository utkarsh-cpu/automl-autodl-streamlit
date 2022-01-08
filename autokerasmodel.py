import imp
import autokeras as ak
from tensorflow import keras

def encoded_data(data_values):
    onehot_encoded=keras.utils.to_categorical(data_values)
    return onehot_encoded

def autokeras_classification(data,values,data_val,value_val,test_data,test_values,no_of_model_to_be_checked,epoch,loss_func,metrics):
    model=ak.StructuredDataClassifier(max_trials=no_of_model_to_be_checked,loss=loss_func,metrics=metrics)
    model.fit(data,encoded_data(values),validation_data=(data_val,encoded_data(value_val)),epochs=epoch)
    predict_y=model.predict(test_data)
    metrics_value=(model.evaluate(test_data,encoded_data(test_values)))
    model_best = model.export_model()
    return predict_y,metrics_value,model_best

def autokeras_regression(data,values,data_val,value_val,test_data,test_values,no_of_model_to_be_checked,epoch,loss_func,metrics):
    model=ak.StructuredDataRegressor(max_trials=no_of_model_to_be_checked,loss=loss_func,metrics=metrics)
    model.fit(data,values,validation_data=(data_val,value_val),epochs=epoch)
    predict_y=model.predict(test_data)
    metrics_value=(model.evaluate(test_data,test_values))
    model_best = model.export_model()
    return predict_y,metrics_value,model_best



