import numpy as np
import torch

my_array = [1,2,3,4,5]

np_arr = np.array(my_array)
np_arr = np.expand_dims(np_arr, axis=0)

print("Numpy array shape: ", np_arr.shape)
tensor = torch.Tensor(my_array)
tensor = torch.unsqueeze(tensor, 0)
print("Tensor shape: ", tensor.shape)
