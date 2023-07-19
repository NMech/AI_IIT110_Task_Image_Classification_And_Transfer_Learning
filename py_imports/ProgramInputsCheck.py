# -*- coding: utf-8 -*-
import os
import logging
from mylogger import mylogger
#%%
logger= mylogger(logging_level = logging.INFO)
#%%
class ProgramInputsCheck:
    
    def __init__(self, res_dir, res_mdl_dataset_1, res_mdl_dataset_2):
        """
        Keyword arguments:\n
            res_dir           : General results filepath.\n
            res_mdl_dataset_1 : Filepath where the model of dataset 1 is to be saved.\n 
            res_mdl_dataset_2 : Filepath where the model of dataset 2 is to be saved.\n 
        """
        self.__create_paths(res_dir)
        self.res_mdl_dataset_1_EXISTS, self.res_mdl_dataset_2_EXISTS, \
        self.res_mdl_dataset_1, self.res_mdl_dataset_2 = self.__check_files_existence(res_mdl_dataset_1, res_mdl_dataset_2)
        
    def __Messages(self, idx):
        """
        Messages generated when the main program's boolean inputs are being checked.\n
        Keyword arguments:\n
            idx : Key index.\n
        """
        msgs = {1:f"{self.res_mdl_dataset_1} DOES NOT EXIST.\n Continuation of Dataset 1 model training is NOT possible.\n FORCED model training from start",
                2:f"{self.res_mdl_dataset_2} DOES NOT EXIST.\n Continuation of Dataset 2 model training is NOT possible.\n FORCED model training from start",
                3:"Dataset 1 model already trained. Training will continue from there",
                4:"Dataset 2 model already trained. Training will continue from there",
                5:"Dataset 1 model training FORCED to start",
                6:"Dataset 2 model training FORCED to start"}
        
        return msgs[idx]
    
    def __create_paths(self, res_dir):
        """
        Creates results paths.\n
        Keyword arguments:\n
            res_dir : General results filepath.\n
        """
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)
        res_dir_mdl          = rf"{res_dir}\models"
        res_dir_metrics      = rf"{res_dir}\metrics"
        res_dir_visuals      = rf"{res_dir}\visuals"
        res_dir_architectures= rf"{res_dir}\architectures"
        res_dir_history      = rf"{res_dir}\history"
        for directory in [res_dir_mdl,res_dir_metrics, res_dir_visuals, res_dir_architectures, res_dir_history]:     
            if not os.path.exists(directory):
                os.mkdir(directory)
        return None

    def __check_files_existence(self, res_mdl_dataset_1, res_mdl_dataset_2):
        """
        Checks if files exists.\n
        Keyword arguments:\n
            res_mdl_dataset_1 : Filepath where the model of dataset 1 is to be saved.\n 
            res_mdl_dataset_2 : Filepath where the model of dataset 2 is to be saved.\n 
        """
        res_mdl_dataset_1_EXISTS, res_mdl_dataset_2_EXISTS = 0, 0
        if os.path.exists(res_mdl_dataset_1) == True:
            res_mdl_dataset_1_EXISTS = 1
        if os.path.exists(res_mdl_dataset_2) == True:
            res_mdl_dataset_2_EXISTS = 1
        return res_mdl_dataset_1_EXISTS, res_mdl_dataset_2_EXISTS, res_mdl_dataset_1, res_mdl_dataset_2
    
    def check_program_inputs(self, bool_train_dataset_1, bool_train_dataset_2, bool_continue_train_dataset_1, bool_continue_train_dataset_2):
        """
        Checks the boolean program inputs and alters them if needed.\n
        Keyword arguments:\n
            bool_train_dataset_1 : Boolean. Train on dataset 1 from start.\n
            bool_train_dataset_2 : Boolean. Train on dataset 2 from start.\n
            bool_continue_train_dataset_1 : Boolean. Continue training on dataset 1.\n
            bool_continue_train_dataset_2 : Boolean. Continue training on dataset 2.\n
        """
        if bool_continue_train_dataset_1 == 1 and self.res_mdl_dataset_1_EXISTS == 0:
            bool_train_dataset_1          = 1
            bool_continue_train_dataset_1 = 0
            logger.warning(self.__Messages(1))
        if bool_continue_train_dataset_2 == 1 and self.res_mdl_dataset_2_EXISTS == 0:
            bool_train_dataset_2          = 1
            bool_continue_train_dataset_2 = 0
            logger.warning(self.__Messages(2))
        if bool_continue_train_dataset_1 == 1 and bool_train_dataset_1 == 1:
            bool_train_dataset_1 = 0
            logger.warning(self.__Messages(3))
        if bool_continue_train_dataset_2 == 1 and bool_train_dataset_2 == 1:
            bool_train_dataset_2 = 0
            logger.warning(self.__Messages(4))
        if bool_continue_train_dataset_1 == 0 and bool_train_dataset_1 == 0 and self.res_mdl_dataset_1_EXISTS == 0:
            bool_train_dataset_1 = 1
            logger.warning(self.__Messages(5))
        if bool_continue_train_dataset_2 == 0 and bool_train_dataset_2 == 0 and self.res_mdl_dataset_2_EXISTS == 0:
            bool_train_dataset_2 = 1
            logger.warning(self.__Messages(6))
            
        return bool_train_dataset_1, bool_train_dataset_2, bool_continue_train_dataset_1, bool_continue_train_dataset_2