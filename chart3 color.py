import matplotlib.pyplot as plt
import numpy as np
ax=plt.gca()
x=np.arange(0,9,1)
y=np.power(x,2)
y2=np.power(x,3)
ax.plot(y,'b--')
ax.plot(y2,'r-.')
ax.set_xlabel('my x label')
ax.set_ylabel('my y label')
ax.set_title('plot title,including $\Omega$')
plt.show()
