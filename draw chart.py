# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 14:24:51 2018

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt
x=np.arange(0,100.5,0.5)
y=2.0*np.sqrt(x)
h=3.0*np.sqrt(x)
plt.plot(x,y)
plt.plot(x,h)
plt.show