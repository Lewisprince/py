import numpy as np
a=np.array([[1,2,1,4,5,6,1,8,9]])
result=np.where(a>2)
print(result)
a[result]=0
print(a)
