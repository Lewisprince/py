import numpy as np
import numpy.linalg as linalg
a= np.matrix([[2,3,-1],[1,3,1],[-2,-2,4]])
b=np.array([1,2,4])
print(a)
print(b)
print(linalg.solve(a,b))
