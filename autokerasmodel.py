import autokeras as ak


def autokeras_classification(data,values,test_data,test_values,no_of_model_to_be_checked,epoch,loss_func,metrics):
    model=ak.StructuredDataClassifier(max_trials=no_of_model_to_be_checked,loss=loss_func,metrics=metrics)
    model.fit(data,values,validation_split=0.3,epochs=epoch)
    predict_y=model.predict(test_data)
    metrics_value=(model.evaluate(test_data,test_values))
    return predict_y,metrics_value

def autokeras_regression(data,values,test_data,test_values,no_of_model_to_be_checked,epoch,loss_func,metrics):
    model=ak.StructuredDataRegressor(max_trials=no_of_model_to_be_checked,loss=loss_func,metrics=metrics)
    model.fit(data,values,validation_split=0.3,epochs=epoch)
    predict_y=model.predict(test_data)
    metrics_value=(model.evaluate(test_data,test_values))
    return predict_y,metrics_value



