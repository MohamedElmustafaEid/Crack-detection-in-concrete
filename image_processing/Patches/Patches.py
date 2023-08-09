# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:28:33 2023

@author: nashw
"""

import numpy as np
import cv2

def extract_patches(image, patch_size, step_size):
    patches = []
    height, width = image.shape[:2]
    
    for y in range(0, height - patch_size[0] + 1, step_size[0]):
        for x in range(0, width - patch_size[1] + 1, step_size[1]):
            patch = image[y:y+patch_size[0], x:x+patch_size[1]]
            patches.append(patch)
    
    return patches

def reconstruct_image(patches, original_shape, patch_size, step_size):
    reconstructed = np.zeros(original_shape, dtype=np.uint8)
    patch_index = 0
    
    for y in range(0, original_shape[0] - patch_size[0] + 1, step_size[0]):
        for x in range(0, original_shape[1] - patch_size[1] + 1, step_size[1]):
            reconstructed[y:y+patch_size[0], x:x+patch_size[1]] = patches[patch_index]
            patch_index += 1
            
    return reconstructed