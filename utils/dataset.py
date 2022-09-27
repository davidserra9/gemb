"""
[1]: https://www.kaggle.com/datasets/nickj26/places2-mit-dataset
[2]: https://www.kaggle.com/datasets/trolukovich/apparel-images-dataset
"""
import os
import cv2
from glob import glob
from os.path import exists, join
from subprocess import run, PIPE
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from prettytable import PrettyTable

def get_training_augmentations():
    """ Function defining and returning the training augmentations.
    Returns
    -------
    train_transform : albumentations.Compose
        training augmentations
    """
    train_transform = [
        A.GaussNoise(p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=5, p=0.2),
            A.Blur(blur_limit=5, p=0.2),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.025, scale_limit=0.1, rotate_limit=10, p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(),
        ], p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        A.Resize(224, 224),
        ToTensorV2(),
    ]
    return A.Compose(train_transform)

def get_validation_augmentations():
    """ Function defining and returning the validation/test augmentations.
        Returns
        -------
        val_transforms : albumentations.Compose
            training augmentations
    """
    val_transforms = [
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        A.Resize(224, 224),
        ToTensorV2(),
    ]
    return A.Compose(val_transforms)

class TripletGUIE(Dataset):

    def __init__(self, root, train, datasets): # Joan: I implemented a more general approach to add new datasets (a list with dataset names)
        """ Dataset for the Google Universal Image Embedding

        Parameters
        ----------
        root: str (path of the root of the datasets)
        train: bool (train or test)
        places: bool (use the places dataset [1])
        apparel: bool (use the apparel-images dataset [2])
        """

        self.root = root
        self.train = train
        self.images = []
        self.labels = []
        self.random_state = np.random.RandomState(29)
        self.transforms = get_training_augmentations() if train else get_validation_augmentations()
        self.code_path = os.getcwd()

        # --- HANDLE PLACES DATASET ---
        if 'places' in datasets:
            # Download the dataset if not is already downloaded
            if not exists(join(self.root, "places2-mit-dataset")):
                print("Downloading places2-mit-dataset...")
                # log = subprocess.check_call("./utils/download_placesdataset.sh '%s'" % {self.root}, shell=True)

                #subprocess.run(f"bash /utils/download_placesdataset.sh {self.root}")

            for subfolder in sorted(
                    glob(join(self.root, "places2-mit-dataset", "train_256_places365standard", "data_256", "*"))):
                for subsubfolder in sorted(glob(join(subfolder, "*"))):
                    crossover = int(0.8 * len(sorted(glob(join(subfolder, "*")))))
                    if train:
                        self.images += sorted(glob(join(subsubfolder, "*")))[:crossover]
                        self.labels += [subsubfolder.split("/")[-1]] * crossover

                    else:
                        self.images += sorted(glob(join(subsubfolder, "*")))[crossover:]
                        self.labels += [subsubfolder.split("/")[-1]] * (
                                len(sorted(glob(join(subsubfolder, "*")))) - crossover)

        # --- HANDLE APPAREL DATASET ---
        if 'apparel' in datasets:
            # Download the dataset if not is already downloaded
            if not exists(join(self.root, "apparel-images-dataset")):
                # run(["bash", f"{join(self.code_path, 'utils', 'download_appareldataset.sh')}", self.root], stdout=PIPE, stderr=PIPE )
                os.system(f"bash {join(self.code_path, 'utils', 'download_appareldataset.sh')} {self.root}")
            else:
                print(f"apparel-images-dataset already downloaded")

            for subfolder in sorted(glob(join(self.root, "apparel-images-dataset", "*"))):
                crossover = int(0.8 * len(sorted(glob(join(subfolder, "*")))))
                if train:
                    self.images += sorted(glob(join(subfolder, "*")))[:crossover]
                    self.labels += [subfolder.split("/")[-1].split("_")[-1]] * crossover

                else:
                    self.images += sorted(glob(join(subfolder, "*")))[crossover:]
                    self.labels += [subfolder.split("/")[-1].split("_")[-1]] * (
                            len(sorted(glob(join(subfolder, "*")))) - crossover)

        # --- HANDLE OBJECTNET DATASET ---
        # REMINDER: train has to be false as ObjectNet dataset cannot be used for training purposes bc of its licence.
        if not train and 'objectnet' in datasets:
            if not exists(join(self.root, "objectnet")):
                os.makedirs(join(self.root, "objectnet"), exist_ok=True)
                os.system(f"bash {join(self.code_path, 'utils', 'download_objectnet.sh')} {join(self.root, 'objectnet')}")

            else:
                print(f"objectnet dataset already downloaded")

            for split_folder in sorted(glob(join(self.root, "objectnet", "*"))):
                for class_folder in sorted(glob(join(split_folder, "*", "images", "*"))):
                    self.images += sorted(glob(join(class_folder, "*")))
                    self.labels += [class_folder.split("/")[-1]] * len(glob(join(class_folder, "*")))

        self.labels2idx = {}
        for idx, label in enumerate(np.unique(self.labels)):
            self.labels2idx[label] = idx

        self.label2positions = {label: np.where(np.asarray(self.labels) == label)[0]
                                for label in self.labels}

        pt = PrettyTable()
        pt.field_names = ['', 'images', 'labels']
        pt.add_row([f"{'train' if train else 'test'} dataset", len(self.images), len(self.labels2idx)])
        print(pt)

        # if not self.train:
        #     self.triplets = [[i,
        #                       self.random_state.choice(self.label2positions[self.labels[i]]),
        #                       self.random_state.choice(self.label2positions[
        #                                                    np.random.choice(
        #                                                        list(set(self.labels) - {self.labels[i]})
        #                                                    )
        #                                                ])
        #                       ]
        #                      for i in range(len(self.images))]

    def __getitem__(self, index):
        if self.train:
            anchor_image, anchor_label = self.images[index], self.labels2idx[self.labels[index]]

            positive_image = self.images[np.random.choice(self.label2positions[self.labels[anchor_label]])]
            positive_label = anchor_label

            negative_pos = np.random.choice(self.label2positions[np.random.choice(list(set(self.labels) - {anchor_label}))])
            negative_image = self.images[negative_pos]
            negative_label = self.labels2idx[self.labels[negative_pos]]

            anchor_image = self.transforms(image=cv2.imread(anchor_image)[:,:,::-1])['image']
            positive_image = self.transforms(image=cv2.imread(positive_image)[:,:,::-1])['image']
            negative_image = self.transforms(image=cv2.imread(negative_image)[:,:,::-1])['image']

            return (anchor_image, positive_image, negative_image), (anchor_label, positive_label, negative_label)

        else:
            # anchor_pos = self.triplets[index][0]
            # anchor_image = self.transforms(image=cv2.imread(self.images[anchor_pos])[:,:,::-1])['image']
            # anchor_label = self.labels2idx[self.labels[anchor_pos]]
            #
            # positive_pos = self.triplets[index][0]
            # positive_image = self.transforms(image=cv2.imread(self.images[positive_pos])[:,:,::-1])['image']
            # positive_label = self.labels2idx[self.labels[positive_pos]]
            #
            # negative_pos = self.triplets[index][0]
            # negative_image = self.transforms(image=cv2.imread(self.images[negative_pos])[:,:,::-1])['image']
            # negative_label = self.labels2idx[self.labels[negative_pos]]

            image = self.transforms(image=cv2.imread(self.images[index])[:, :, ::-1])['image']
            label = self.labels2idx[self.labels[index]]
            return image, label

        # Joan: TODO:Perform here the transform

    def __len__(self):
        return len(self.images)  # if you want to subsample for speed


if __name__ == "__main__":
    dataset = TripletGUIE(root="/home/david/Workspace/gemb/data",
                          train=True,
                          places=False,
                          apparel=True)

    (img1, img2, img3), (l1, l2, l3) = dataset[0]