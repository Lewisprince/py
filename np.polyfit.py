import numpy as np
x=np.array([0.0,1.0,2.0,3.0,4.0,5.0])
y=np.array([0.0,0.8,0.9,0.1,-0.8,-1.0])
coef=np.polyfit(x,y,1)
p=np.poly1d(coef)
print(p)

