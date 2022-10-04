import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import open_clip
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import TripletGUIE, get_validation_augmentations


class Shared_CLIP_MLP(nn.Module):
    def __init__(self, clip_model='ViT-B-16-plus-240', pretrained='laion400m_e32'):
        super().__init__()
        self.backbone_clip, _, _ = open_clip.create_model_and_transforms(model_name=clip_model,
                                                                         pretrained=pretrained)

        for param in self.backbone_clip.parameters():
            param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, img):
        x = self.backbone_clip.encode_image(img)
        x = self.mlp(x)
        return x

class Triplet_CLIP_MLP(nn.Module):
    def __init__(self, clip_model='ViT-H-14', pretrained='laion2b_s32b_b79k'):
        super().__init__()
        self.shared_embedding = Shared_CLIP_MLP(clip_model=clip_model, pretrained=pretrained)

    def forward(self, anchor, positive, negative):
        return self.shared_embedding(anchor), self.shared_embedding(positive), self.shared_embedding(negative)

    def embedding(self, img):
        return self.shared_embedding(img)

if __name__ == "__main__":
    model = Triplet_CLIP_MLP(clip_model='ViT-B-32', pretrained='openai')
    dataset = TripletGUIE(root="/home/david/Workspace/gemb/data",
                          train=True,
                          places=False,
                          apparel=True)

    (img1, img2, img3), (l1, l2, l3) = dataset[0]

    out = model(img1.unsqueeze(0))
    print(out.shape)

