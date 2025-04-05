import torch
from torch.utils.data import Dataset
import torchio as tio


"""
Klasa której celem jest przygotowanie i pobranie odpowiednich danych dla DataLoadera
"""
class SegmentationDataset(Dataset):

    """
    image_paths - lista ścieżek do plików z obrazami 3D
    mask_paths - lista ścieżek do plików z maskami
    transform - opcjonalne augmentacje
    """
    def __init__(self,image_paths,mask_paths,target_shape = None ,transform=None):

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

        if target_shape:
            shape_transform = tio.CropOrPad(target_shape)
            if transform:
                self.transform = tio.Compose([shape_transform,transform])
            else:
                self.transform = shape_transform
        

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):

        image = tio.ScalarImage(self.image_paths[index])
        mask = tio.LabelMap(self.mask_paths[index])

        subject = tio.Subject(image=image,mask=mask)

        if self.transform:
            subject = self.transform(subject)

        image_tensor = subject['image'].data
        mask_tensor = subject['image'].data

        return image_tensor, mask_tensor