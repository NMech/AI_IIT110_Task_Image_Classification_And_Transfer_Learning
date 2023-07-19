# -*- coding: utf-8 -*-
from io import StringIO
#%%
def message_generation(dict_key, dynamic_display="", dataset_name=""):
    """
    This function contains a dictionary with the messages that are displayed in main's\n
    program terminal and subsequently saved in the log file.\n
    Keyword arguments:\n
        dict_key        : Key of the dictionary.\n
        dynamic_display : Extra message dynamically changed from main program.\n
    Returns a string.\n
    """
    Message = {"0a" : "GPU used for models training",
               "0b" : "CPU used for models training",   
               "1a" : f"Processing {dataset_name} images. Time elapsed : {dynamic_display}",
               "1b" : f"Loading processed {dataset_name} images. Time elapsed : {dynamic_display}",
               "2"  : f"Creating train/validation datasets for {dataset_name}. Time elapsed : {dynamic_display}",
               "3a" : f"Continue training {dataset_name} model. Time elapsed : {dynamic_display}",
               "3ae": f"Continue training {dataset_name} model. Exception occured: {dynamic_display}",
               "3b" : f"Training {dataset_name}model. Time elapsed : {dynamic_display}",
               "3c" : f"Model summary: {dynamic_display}",
               "3d" : f"{dataset_name} model already trained",
               "3be": f"Training model exception occured: {dynamic_display}",
               "4"  : f"Metrics of {dataset_name} images. Time elapsed : {dynamic_display}",
               "8"  : f"Total time elapsed : {dynamic_display}"
              } 

    return Message[dict_key]

def get_model_summary(model):
    """
    Function used for getting the model summary (tensorflow) and returning it as a string.\n
    Keyword arguments:\n
        model : keras.tensorflow.model.\n
    Returns a string.\n
    """
    summary_string = StringIO()
    model.summary(print_fn=lambda x: summary_string.write(x + "\n"), show_trainable=True)

    return summary_string.getvalue()