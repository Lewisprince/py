import numpy as np
import matplotlib.pyplot as plt
data=np.random.randn(1000)
ax1=plt.subplot(1,2,1)
ax2=plt.subplot(1,2,2)
ax1.hist(data,bins=30,color='b')
ax2.hist(data,bins=69,color='r')
plt.show()
