import autokeras as ak

def autokeras_classification(data,values,val_data,val_values,no_of_model_to_be_checked):
    model=ak.StructuredDataClassifier(max_trials=no_of_model_to_be_checked)

