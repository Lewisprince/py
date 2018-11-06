import numpy as np
import matplotlib.pyplot as plt
array1=[0.3,0.2,0.26,0.4,0.36,0.4]
array2=[0.4,0.13,0.2,0.31,0.33,0.27]
array3=[0.25,0.4,0.39,0.28,0.26,0.19]
#samp1 = np.random.normal(loc=0., scale=1.,size=100)  # normal 正态分布
#samp2 = np.random.normal(loc=1., scale=2., size=100)
#samp3 = np.random.normal(loc=0.3, scale=1.2, size=100)
ax = plt.gca()
ax.boxplot((array1, array2, array3))
ax.set_xticklabels(['sample 1', 'sample 2', 'sample 3'])

plt.show()
