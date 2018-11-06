import numpy as np
a=np.arange(0,100)*0.5
b=np.arange(-100,0)*0.5
np.save('a-file',a)
np.save('b-file',b)
np.savez('ab-file',a=a,b=b)
