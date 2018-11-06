# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 14:42:20 2018

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt
x=np.arange(0,100.5,0.5)
y=2.0*np.sqrt(x)
fig=plt.figure()
ax=fig.add_axes([0.6,0.6,0.8,0.8])
ax.plot(x,y)
ax.set_yticks([0,1,10,50,100])
ax.set_ylabel('Y/m',size='xx-large')
plt.title('oops',size='xx-large')
plt.show()