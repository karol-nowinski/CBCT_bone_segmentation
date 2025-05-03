import torch
import torchio as tio
from pathlib import Path
import config
from torch.utils.data import DataLoader



class UnetInference:
    def __init__(self,model,device = None,patch_size = (128,128,128),patch_overlap = (64,64,64),batch_size = 1):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.batch_size = batch_size
        self.model.to(self.device)
        self.model.eval()


    def predict_subject(self,subject):

        sampler = tio.inference.GridSampler(subject=subject,patch_size=self.patch_size,patch_overlap=self.patch_overlap)
        aggregator = tio.inference.GridAggregator(sampler=sampler)
        loader = DataLoader(sampler,batch_size=self.batch_size)

        with torch.no_grad():
            print("no grad")
            total_batches = str(len(loader))
            for idx, batch in enumerate(loader):
                print(f"⏳ Patch {str(idx+1)}/{total_batches}", end='\r')
                input = batch['image'][tio.DATA].to(self.device)
                location = batch[tio.LOCATION]
                prediction = self.model(input)
                if self.model.deep_supervision == True:
                    aggregator.add_batch(prediction[-1].cpu(),locations=location)
                else:
                    aggregator.add_batch(prediction.cpu(),locations=location)

        return aggregator.get_output_tensor()
    
    def predict_and_save(self,image_path,output_path,save=True):

        subject = tio.Subject(
            image = tio.ScalarImage(str(image_path))
        )

        output_tensor = self.predict_subject(subject)

        segmentation = output_tensor.argmax(dim=0, keepdim=True)

        print("Output tensor shape:", output_tensor.shape)
        _, D, H, W = output_tensor.shape  # zakładamy (C, D, H, W)
        print("🔍 example logits (środek):", output_tensor[:, D//2, H//2, W//2])
        print("Unikalne klasy po argmax:", segmentation.unique())

        if save:
            label_map = tio.LabelMap(
                tensor=segmentation,
                affine=subject.image.affine
            )
            label_map.save(output_path)

        pass