import numpy as np
import torch

def numpy_to_tensor(ct_data):
    """
    Converting 3D ct numpy array to tensor normalized as [0.0,1.0]
    args:
        [height, width, depth, channels]
    """
    ct_data = torch.from_numpy(ct_data)
    for idx in range(ct_data.shape[-1]):
        ct_data[...,idx] = (ct_data[...,idx] - torch.min(ct_data[...,idx])) / (torch.max(ct_data[...,idx]) - torch.min(ct_data[...,idx]))
    return ct_data

predicted_mask = np.zeros((3,512,512,512))
print(predicted_mask.shape)

predicted_mask = np.moveaxis(predicted_mask, 0, -1)
print(predicted_mask.shape)
predicted_mask = numpy_to_tensor(predicted_mask)
print(predicted_mask.shape)