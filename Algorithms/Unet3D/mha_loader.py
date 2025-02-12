from torch.utils.data import Dataset, DataLoader
import torch
import SimpleITK as sitk
import numpy as np


# def load_mha(filepath):
#     image = sitk.ReadImage(filepath)  # Read .mha file
#     padded_image = pad_image
#     array = sitk.GetArrayFromImage(image)  # Convert to NumPy array (Z, Y, X)
#     return array

class MHA_Dataset(Dataset):
    def __init__(self, image_paths, label_paths,target_shape, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.target_shape = target_shape
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.load_mha(self.image_paths[idx])
        label = self.load_mha(self.label_paths[idx])

        # Convert to PyTorch tensors and normalize
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (C, Z, Y, X)
        label = torch.tensor(label, dtype=torch.long)  # (Z, Y, X)

        if self.transform:
            image, label = self.transform(image, label)

        return image, label
    
    def load_mha(self, filepath):
        image = sitk.ReadImage(filepath)  # Read .mha file
        array = sitk.GetArrayFromImage(image)  # Convert to NumPy array (Z, Y, X)
        padded_image = pad_image(array, self.target_shape)
        #array_padded = sitk.GetArrayFromImage(padded_image)
        return padded_image
    
def pad_image(image,target_shape):
    padding = [
        (0, target_shape[0] - image.shape[0]),  # Z padding
        (0, target_shape[1] - image.shape[1]),  # Y padding
        (0, target_shape[2] - image.shape[2])   # X padding
    ]

    padded_image = np.pad(image,padding,mode='constant',constant_values=0)
    return padded_image