import numpy as np
a = np.array([[1,2], [3,4]])
a = np.mean(a, axis=0, keepdims=True)
print(a)