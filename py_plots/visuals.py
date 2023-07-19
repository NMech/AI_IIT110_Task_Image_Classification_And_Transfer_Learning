# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
#%%
# =============================================================================
# def visual_chestXray14(df_original, df_binary, save_dir, img_name):
#     """
#     Keyword arguments:\n
#         df_original :
#         df_binary   :
#     """
#     class_counts = df_original.sum()
#     #bins = np.arange(min(df_original), max(df_original) + 1.5) - 0.5
#     fig, axs = plt.subplots(2, 1)
#     #axs[0].hist(df_original, bins=bins)
#     axs[0].bar(class_counts.index, class_counts.values, width=1.0, align='center', edgecolor="black")
#     axs[0].set_ylabel("Count")
#     axs[0].tick_params(axis="x", rotation=45)
#     axs[0].grid(True, axis="y")
#     
#     bins   = [-0.5, 0.5,  1.5]
#     bins_c = [0, 1]
#     axs[1].hist(df_binary, bins=bins, edgecolor="black")
#     axs[1].set_xticks(bins_c, labels=["No Finding", "Finding"])
#     axs[1].grid(True, axis="y")
#     fig.tight_layout()
#     fig.savefig(rf"{save_dir}\{img_name}")
#                 
#     return None
# =============================================================================

# =============================================================================
# def visual_MURA(df_binary, save_dir, img_name):
#     """
#     Keyword arguments:\n
#         df_binary :
#     """
#     fig, ax = plt.subplots()
#     bins   = [-0.5, 0.5,  1.5]
#     bins_c = [0, 1]
#     ax.hist(df_binary, bins=bins, edgecolor="black")
#     ax.set_xticks(bins_c, labels=["No Finding", "Finding"])
#     ax.grid(True, axis="y")
#     fig.tight_layout()
#     fig.savefig(rf"{save_dir}\{img_name}")
#     
#     return None
# =============================================================================
    
def visual_image(image_array, image_label="", save_dir="", img_name=""):
    """
    Function used for plotting a sampling of the processed ChestXray14 images.\n
    Keyword arguments:\n
        image_array :\n
        save_dir    : Saving directory.\n
        img_name    :\n
    """
    fig, ax = plt.subplots()
    ax.imshow(image_array, cmap="gray", vmin=0, vmax=255)
    ax.set_title(image_label)
    if save_dir != "" and img_name != "":
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(rf"{save_dir}\{img_name}", bbox_inches="tight", pad_inches=0)

    return None

# =============================================================================
# def visual_image_rgb(image_array, mean, std):
#     """
#     Function used for plotting a single image.
#     Args:
#         image_array: The image array to be plotted (shape: (height, width, channels)).
#         mean: The mean used for normalization (shape: (1, 1, 3)).
#         std: The standard deviation used for normalization (shape: (1, 1, 3)).
#     """
#     # Reverse the normalization by scaling the pixel values for each channel
#     image_array = (image_array * std) + mean
#     
#     # Clip the values to the valid range [0, 255]
#     image_array = np.clip(image_array, 0, 255)
#     
#     # Convert the image array to unsigned integers (0-255)
#     image_array = np.uint8(image_array)
#     
#     # Plot the image
#     plt.imshow(image_array)
#     plt.axis('off')
#     plt.show()
# =============================================================================
