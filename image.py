# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from PIL import Image
import numpy as np

im=np.array(Image.open('C:\\Users\DELL\Pictures\IMG_3080.JPG'))
print(im.shape,im.dtype)

