# -*- coding: utf-8 -*-
import os
import logging
import h5py
import cv2
import re
import numpy as np
import pandas as pd
from PIL      import Image
from mylogger import mylogger
#%%
logger = mylogger(logging_level = logging.INFO)
#%%
class FOOD101_UECFOOD256_Preprocess:
    
    def __init__(self, FOOD_dataset_DIR, resized_size_img):
        """
        FOOD101 OR UECFOOD256 dataset preprocessing initialization.\n
        Keyword arguments:\n
            FOOD_dataset_DIR : General filepath of FOOD101 OR UECFOOD256 dataset.\n
            resized_size_img : tuple of the shape of the input images. (nx_pixels, ny_pixels, 1).\n
        """
        self.FOOD_dataset_DIR = FOOD_dataset_DIR
        self.resized_size_img = resized_size_img[:2]
        self.FOOD_dataset_DIR_images       = rf"{self.FOOD_dataset_DIR}\images"
        self.FOOD_dataset_DIR_images_saved = rf"{self.FOOD_dataset_DIR}\images_saved_{self.resized_size_img}"

    def __save_images(self, Images, save_dir, saved_filename):
        """
        Auxiliary function used for saving the FOOD101 OR UECFOOD256 dataset images in .h5 files.\n
        """ 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        os.chdir(save_dir)
    
        with h5py.File(f"{save_dir}\{saved_filename}.hdf5", "w") as fOut:
            for i, arr in enumerate(Images):
                fOut.create_dataset(f"array_{i}", data=arr)
        
        return None
    
    def __resize_image(self, image):
        """
        Auxiliary function used for resizing images.\n
        Keyword arguments:\n
            image : mp.array representation of an image.\n
        Returns a np.array (resized image).\n
        """
        image_resized = cv2.resize(image, self.resized_size_img)
        return image_resized

    def save_images(self):
        """
        Save images in h5 files.\n
        """
        bounding_boxes_exist = False
        food_directories = os.listdir(self.FOOD_dataset_DIR_images)
        for food_dir in food_directories:
            logger.info(f"Processing files of dir: {food_dir}")
            Images = []
            full_food_dir = os.path.join(self.FOOD_dataset_DIR_images, food_dir)
            images = [file for file in os.listdir(full_food_dir) if file.endswith(".jpg")]
            if "bb_info.txt" in os.listdir(full_food_dir):
                bounding_boxes = pd.read_csv(os.path.join(full_food_dir, "bb_info.txt"), index_col="img", sep = " ")
                bounding_boxes_exist = True
            file_cntr, img_cntr = 1, 0
            for img in images:
                
                image_path  = os.path.join(full_food_dir, img)
                image       = Image.open(image_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                if bounding_boxes_exist == True:
                    idx = int(img[:-4])
                    try:
                        image = image.crop((bounding_boxes.loc[idx].x1, bounding_boxes.loc[idx].y1, bounding_boxes.loc[idx].x2, bounding_boxes.loc[idx].y2))
                    except Exception:
                        pass
                image_array = np.asarray(image)
                image_array = self.__resize_image(image_array)
                Images.append( image_array )
                
                if img_cntr%1000 == 0 or img_cntr == len(images)-1:
                    saved_filename = f"{food_dir}_{file_cntr:02}"
                    if img_cntr != 0:
                        logger.info(f"Processed {img_cntr+1} out of {len(images)}")
                        self.__save_images(Images, self.FOOD_dataset_DIR_images_saved, saved_filename)
                        
                        file_cntr += 1
                img_cntr +=1
                
        return None
    
    def read_images(self, keep_imgs=0.3):
        """
        Read images from h5 files.\n
        """
        Img_labels = []
        local_idx = 1
        files = os.listdir(self.FOOD_dataset_DIR_images_saved)

        for idx, file in enumerate(files):
            logger.info(f"Reading processed file {idx+1:3} : {file}")
            full_h5_filename = os.path.join(self.FOOD_dataset_DIR_images_saved, file)
            with h5py.File(full_h5_filename, "r") as f:
                keys = list(f.keys())
                images = np.array([f[key][()] for key in keys])
                imgs_kept = int(keep_imgs*len(images))
                imgs_idxs = np.random.randint(0, len(images), imgs_kept)
                if local_idx == 1:
                    Img_matrix = images[imgs_idxs]
                    Img_labels = [re.sub(r"_\d+\.hdf5", "", file)]*imgs_kept
                else:
                    Img_matrix = np.concatenate((Img_matrix, images[imgs_idxs]))
                    Img_labels += [re.sub(r"_\d+\.hdf5", "", file)]*imgs_kept
            local_idx += 1
 
        return Img_matrix, Img_labels