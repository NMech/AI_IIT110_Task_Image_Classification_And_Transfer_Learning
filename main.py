# -*- coding: utf-8 -*-
import os
import sys
import logging
import time
import tensorflow as tf
import numpy      as np
ROOT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT_DIR, "py_imports"))
sys.path.append(os.path.join(ROOT_DIR, "py_plots"))
#%%
from messages                      import message_generation, get_model_summary
from mylogger                      import mylogger
from NeuralNetworks                import NeuralNetworks
from PostProcess                   import PostProcess
from Preprocess_FOOD101_UECFOOD256 import FOOD101_UECFOOD256_Preprocess
from ProgramInputsCheck            import ProgramInputsCheck
from ClassificationMetricsPlots    import ClassificationMetricsPlot
#%%
from sklearn.metrics            import confusion_matrix
from tensorflow.keras.models    import load_model
from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import OneHotEncoder
#%%
start_0       = time.perf_counter()
logging_Level = logging.INFO
logger        = mylogger(logging_level = logging_Level)
#%%
gpu_devices = tf.config.list_physical_devices("GPU")
if len(gpu_devices) > 0:
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)  
    logger.info(message_generation("0α"))
else:
    logger.info(message_generation("0b"))
#%%
################################################################################
#################################### Inputs ####################################
################################################################################
select_model                   = 4
model_optimizer                = "adadelta"
nEpochs                        = 1
batch_Size                     = 64
bool_preprocess_imgs_dataset_1 = 1
bool_preprocess_imgs_dataset_2 = 1
bool_train_dataset_1           = 1
bool_train_dataset_2           = 1
bool_continue_train_dataset_1  = 0
bool_continue_train_dataset_2  = 0
resized_input_images           = (256, 256, 3)
testSize                       = 0.15
valSize                        = 0.15
unfreeze_layers_original       = 20
unfreeze_layers_transfer       = 0 # for transfer learning (6 for 3 conv layers, 14 for all 6 additional conv layers)
img_generator                  = "generator2"
regularizer                    = "no" # adding regularizer in dense layers
#%%
dataset_1_name = "UECFOOD256"
dataset_2_name = "FOOD101"
if dataset_1_name == "FOOD101" and dataset_2_name == "UECFOOD256":
    imgs_kept_1 = 0.4
    imgs_kept_2 = 1.0
    p1, p2 = 1 , 0
else:
    imgs_kept_1 = 1.0
    imgs_kept_2 = 0.4
    p1, p2 = 0 , 1
models_selection = {1:"C15D3net", 2:"VGG16mod", 3:"VGG19mod", 4:"ResNet50mod"}
#%%
model_selector         = models_selection[select_model]
target_size            = resized_input_images[:2]
common_tag1            = f"{target_size}_{model_selector}_generator_{img_generator[-1]}_regularizer_{regularizer}_nlu1_{unfreeze_layers_original}"
common_tag2            = f"{common_tag1}_nlu2_{unfreeze_layers_transfer}"
#%%
dataset_1_DIR          = f"{ROOT_DIR}\dummy_data\{dataset_1_name}"
dataset_2_DIR          = f"{ROOT_DIR}\dummy_data\{dataset_2_name}"
res_dir                = rf"{ROOT_DIR}\results"
res_dir_mdl            = rf"{res_dir}\models"
res_dir_metrics        = rf"{res_dir}\metrics"
res_dir_visuals        = rf"{res_dir}\visuals"
res_dir_architectures  = rf"{res_dir}\architectures"
res_dir_history        = rf"{res_dir}\history"
tensorboard_dir        = rf"{ROOT_DIR}\log_tensorboard"
#%%
mdl_dataset_1          = f"model_dataset_1_{common_tag1}"
res_mdl_dataset_1      = rf"{res_dir_mdl}\{mdl_dataset_1}.h5"
res_metrics_dataset_1  = rf"{res_dir_metrics}\{mdl_dataset_1}.dat"
res_archit_dataset_1   = rf"{res_dir_architectures}\{mdl_dataset_1}.dat"
csv_logger_dataset_1   = rf"{ROOT_DIR}\logs\training_log_dataset_1_{common_tag1}.csv"
res_history_dataset_1  = rf"{res_dir_history}\history_dataset_1_{common_tag1}.csv"
conf_matrix_dataset_1  = f"conf_matrix_dataset_1_{common_tag1}.pdf"
model_name_dataset_1   = f"Dataset_1_{common_tag1.replace('(', '').replace(')', '').replace(', ', '_')}"
#%%
mdl_dataset_2          = f"model_dataset_2_{common_tag2}"
res_mdl_dataset_2      = rf"{res_dir_mdl}\{mdl_dataset_2}.h5"
res_metrics_dataset_2  = rf"{res_dir_metrics}\{mdl_dataset_2}.dat"
res_archit_dataset_2   = rf"{res_dir_architectures}\{mdl_dataset_2}.dat"
csv_logger_dataset_2   = rf"{ROOT_DIR}\logs\training_log_dataset_2_{common_tag2}.csv"
res_history_dataset_2  = rf"{res_dir_history}\history_dataset_2_{common_tag2}.csv"
conf_matrix_dataset_2  = f"conf_matrix_dataset_2_{common_tag2}.pdf"
model_name_dataset_2   = f"Dataset_2_{common_tag2.replace('(', '').replace(')', '').replace(', ', '_')}"
#%%
res_imgs_preprocessed  = rf"{res_dir}\preprocessed_images\{resized_input_images[0]}"
#%%
prgrmcheckOBJ = ProgramInputsCheck(res_dir, res_mdl_dataset_1, res_mdl_dataset_2)
bool_train_dataset_1, bool_train_dataset_2, bool_continue_train_dataset_1, bool_continue_train_dataset_2 = \
prgrmcheckOBJ.check_program_inputs(bool_train_dataset_1, bool_train_dataset_2, bool_continue_train_dataset_1, bool_continue_train_dataset_2)
#%%
################################################################################
########################### Reading dataset 1 images ###########################
################################################################################
PreProcess_dataset1OBJ = FOOD101_UECFOOD256_Preprocess(dataset_1_DIR, resized_input_images)
if bool_preprocess_imgs_dataset_1 == True:
    start_time  = time.perf_counter()
    PreProcess_dataset1OBJ.save_images()
    finish_time = time.perf_counter()
    time_diff   = f"{round((finish_time-start_time)/60, 2)} minutes(s)"
    logger.info(message_generation("1a", time_diff, dataset_1_name))

start_time = time.perf_counter()
x_data1, y_data1 = PreProcess_dataset1OBJ.read_images(imgs_kept_1)
nclasses_1   = len(np.unique(y_data1))
y_data1      = np.array(y_data1).reshape(-1, 1)
y_data1      = OneHotEncoder(sparse_output=False).fit_transform(y_data1)
finish_time  = time.perf_counter()
time_diff    = f"{round((finish_time-start_time)/60, 2)} minutes(s)"
logger.info(message_generation("1b", time_diff, dataset_1_name))
#%%
################################################################################
################## Splitting dataset 1 images (train/val/test) #################
################################################################################
start_time = time.perf_counter()
x_trainval, x_test, y_trainval, y_test = train_test_split(x_data1, y_data1, test_size=testSize, random_state=42)
x_train, x_val, y_train, y_val         = train_test_split(x_trainval, y_trainval, test_size=(1.-testSize)*valSize, random_state=42)               
finish_time = time.perf_counter()
time_diff = f"{round((finish_time-start_time)/60, 2)} minutes(s)"
logger.info(message_generation("2", time_diff, dataset_1_name))
#%%
################################################################################
####################### Train model for dataset 1 images #######################
################################################################################
DIRS  = [res_mdl_dataset_1, csv_logger_dataset_1, res_history_dataset_1, res_archit_dataset_1, tensorboard_dir ]
nnOBJ = NeuralNetworks(nclasses_1, model_selector, resized_input_images, DIRS, img_generator, **{"model_optim":model_optimizer, "nEpochs":nEpochs, "batch_size":batch_Size, "model_regul":regularizer})
XY    = (x_train, y_train, x_val, y_val)
if bool_continue_train_dataset_1 == True:
    try:
        start_time = time.perf_counter()
        model_dataset_1, datagen = nnOBJ.train_model(XY, model_name_dataset_1, "load", unfreeze_layers_original)
        finish_time = time.perf_counter()
        time_diff    = f"{round((finish_time-start_time)/60, 2)} minutes(s)"
        logger.info(message_generation("3a", time_diff, dataset_1_name))
        logger.info(message_generation("3c", get_model_summary(model_dataset_1), dataset_1_name))
    except Exception as ex:
        logger.info(message_generation("3ae", ex))
      
elif bool_train_dataset_1 == True:
    try:
        start_time = time.perf_counter()
        model_dataset_1, datagen   = nnOBJ.train_model(XY, model_name_dataset_1, "compile", unfreeze_layers_original )
        finish_time = time.perf_counter()
        time_diff    = f"{round((finish_time-start_time)/60, 2)} minutes(s)"
        logger.info(message_generation("3b", time_diff, dataset_1_name))
        logger.info(message_generation("3c", get_model_summary(model_dataset_1), dataset_1_name))
    except Exception as ex:
        logger.error(message_generation("3be", ex, dataset_1_name))
else:
    model_dataset_1 = load_model(res_mdl_dataset_1)
    logger.info(message_generation("3d", "", dataset_1_name)) 
#%%
#-----------------------------------------------------------------------------#
######################## Second part: Transfer Learning ########################
#-----------------------------------------------------------------------------#  
#%%
################################################################################
########################### Reading dataset 2 images ###########################
################################################################################
start_time = time.perf_counter()
PreProcess_dataset2OBJ = FOOD101_UECFOOD256_Preprocess(dataset_2_DIR, resized_input_images)
if bool_preprocess_imgs_dataset_2 == True:
    start_time  = time.perf_counter()
    PreProcess_dataset2OBJ.save_images()
    finish_time = time.perf_counter()
    time_diff    = f"{round((finish_time-start_time)/60, 2)} minutes(s)"
    logger.info(message_generation("1a", time_diff, dataset_2_name))

start_time = time.perf_counter()
x_data2, y_data2 = PreProcess_dataset2OBJ.read_images(imgs_kept_2)
nclasses_2= len(np.unique(y_data2))
y_data2 = np.array(y_data2).reshape(-1, 1)
y_data2 = OneHotEncoder(sparse_output=False).fit_transform(y_data2)
finish_time = time.perf_counter()
time_diff = f"{round((finish_time-start_time)/60, 2)} minutes(s)"
logger.info(message_generation("1b", time_diff, dataset_2_name))
#%%
################################################################################
################### Splitting dataset 2 images (train/val/test) ##################
################################################################################
start_time = time.perf_counter()
x_trainval_2, x_test_2, y_trainval_2, y_test_2 = train_test_split(x_data2, y_data2, test_size=testSize, random_state=42)
x_train_2, x_val_2, y_train_2, y_val_2         = train_test_split(x_trainval_2, y_trainval_2, test_size=(1.-testSize)*valSize, random_state=42)               
finish_time = time.perf_counter()
time_diff = f"{round((finish_time-start_time)/60, 2)} minutes(s)"
logger.info(message_generation("2", time_diff, dataset_2_name))
#%%
################################################################################
####################### Train model for dataset 2 images #######################
################################################################################
DIRS  = [res_mdl_dataset_2, csv_logger_dataset_2, res_history_dataset_2, res_archit_dataset_2, tensorboard_dir ]
nnOBJ = NeuralNetworks(nclasses_2, model_selector, resized_input_images, DIRS, **{"model_optim":model_optimizer, "nEpochs":nEpochs, "batch_size":batch_Size})
XY_2  = (x_train_2, y_train_2, x_val_2, y_val_2)
if bool_continue_train_dataset_2 == True:
    try:
        start_time = time.perf_counter()
        model_dataset_2, datagen_2 = nnOBJ.train_transfer_learning_model(XY_2, model_name_dataset_2, "load", res_mdl_dataset_1, unfreeze_layers_original, unfreeze_layers_transfer)
        finish_time = time.perf_counter()
        time_diff    = f"{round((finish_time-start_time)/60, 2)} minutes(s)"
        logger.info(message_generation("3a", time_diff, dataset_2_name))
        logger.info(message_generation("3c", get_model_summary(model_dataset_2), dataset_2_name))
    except Exception as ex:
        logger.info(message_generation("3ae", ex))        
 
elif bool_train_dataset_2 == True:
    try:
        start_time = time.perf_counter()
        model_dataset_2, datagen_2 = nnOBJ.train_transfer_learning_model(XY_2, model_name_dataset_2, "compile", res_mdl_dataset_1, unfreeze_layers_original, unfreeze_layers_transfer)
        finish_time = time.perf_counter()
        time_diff = f"{round((finish_time-start_time)/60, 2)} minutes(s)"
        logger.info(message_generation("3b", time_diff, dataset_2_name))
        logger.info(message_generation("3c", get_model_summary(model_dataset_2), dataset_2_name))
    except Exception as ex:
        logger.error(message_generation("3be", ex, dataset_2_name))
else:
    model_dataset_2 = load_model(res_mdl_dataset_2)
    logger.info(message_generation("3c", "", dataset_2_name))
#%%
################################################################################
##################### Calculation & Presentation of results ####################
################################################################################
start_time = time.perf_counter()
yPred_proba_dataset_1 = model_dataset_1.predict(np.array(x_test))
yPred_dataset_1       = np.argmax(yPred_proba_dataset_1,axis=1)
y_test_1_asarray      = np.argmax(y_test, axis=1)

if nclasses_1 >= 10: annot = False 
else: annot = True

plotOBJ_dataset_1     = ClassificationMetricsPlot(y_test_1_asarray)
CMat_dataset_1 = confusion_matrix(y_test_1_asarray, yPred_dataset_1)
plotOBJ_dataset_1.Confusion_Matrix_Plot(yPred_dataset_1, CMat_dataset_1, normalize=True, labels="auto", 
                    cMap="default", Title=f"Confusion Matrix {dataset_1_name} Test dataset",
                    Rotations=[0.,0.], annotation=annot, savePlot=[True, res_dir_visuals, conf_matrix_dataset_1])

postOBJ_dataset_1 = PostProcess(yPred_dataset_1,yPred_proba_dataset_1, y_test_1_asarray, res_metrics_dataset_1)
postOBJ_dataset_1.calculate_multiclass_metrics()
finish_time = time.perf_counter()
time_diff   = f"{round((finish_time-start_time)/60, 2)} minutes(s)"
logger.info(message_generation("4", time_diff, dataset_1_name))
#%%
start_time = time.perf_counter()
yPred_proba_dataset_2 = model_dataset_2.predict(np.array(x_test_2))
yPred_dataset_2       = np.argmax(yPred_proba_dataset_2, axis=1)
y_test_2_asarray      = np.argmax(y_test_2, axis=1)

if nclasses_2 >= 10: annot = False 
else: annot = True

plotOBJ_dataset_2     = ClassificationMetricsPlot(y_test_2_asarray)
CMat_dataset_2 = confusion_matrix(y_test_2_asarray, yPred_dataset_2)
plotOBJ_dataset_2.Confusion_Matrix_Plot(yPred_dataset_2, CMat_dataset_2, normalize=True, labels="auto", 
                    cMap="default", Title=f"Confusion Matrix {dataset_2_name} Test dataset",
                    Rotations=[0.,0.], annotation=annot, savePlot=[True, res_dir_visuals, conf_matrix_dataset_2])

postOBJ_dataset_2 = PostProcess(yPred_dataset_2, yPred_proba_dataset_2, y_test_2_asarray, res_metrics_dataset_2)
postOBJ_dataset_2.calculate_multiclass_metrics()
finish_time = time.perf_counter()
time_diff   = f"{round((finish_time-start_time)/60, 2)} minutes(s)"
logger.info(message_generation("4", time_diff, dataset_2_name))