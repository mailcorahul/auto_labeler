import numpy as np
import matplotlib.pyplot as plt
import gc
import cv2

def draw_mask(image, mask_generated) :
  masked_image = image.copy()

  masked_image = np.where(mask_generated.astype(int),
                          np.array([0,255,0], dtype='uint8'),
                          masked_image)

  masked_image = masked_image.astype(np.uint8)

  return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)