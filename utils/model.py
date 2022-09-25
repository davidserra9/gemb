import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import open_clip
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import TripletGUIE, get_validation_augmentations

class CLIP_MLP(nn.Module):
    def __init__(self, clip_model='ViT-g-14'):
        super().__init__()
        self.backbone_clip, _, preprocess = open_clip.create_model_and_transforms(model_name=clip_model,
                                                                                  pretrained='openai')
        #TODO MLP (LA SALIDA DEL CLIP TIENE 512 PARAMS - torch.Size([1, 512]))
        self.size = 224
    def forward(self, img):

        x = T.functional.resize(img, size=[self.size, self.size])
        x = x / 255.0
        x = T.functional.normalize(x,
                                   mean=[0.48145466, 0.4578275, 0.40821073],
                                   std=[0.26862954, 0.26130258, 0.27577711])

        out = self.backbone_clip.encode_image(x)
        return out

if __name__ == "__main__":
    model = CLIP_MLP(clip_model='ViT-B-32')
    dataset = TripletGUIE(root="/home/david/Workspace/gemb/data",
                          train=True,
                          places=False,
                          apparel=True)

    (img1, img2, img3), (l1, l2, l3) = dataset[0]


    out = model(img1.unsqueeze(0))
    print(out.shape)

