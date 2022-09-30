"""
[1]: https://www.kaggle.com/datasets/nickj26/places2-mit-dataset
[2]: https://www.kaggle.com/datasets/trolukovich/apparel-images-dataset
"""
import os
import cv2
from glob import glob
from tqdm import tqdm
from os.path import exists, join
from subprocess import run, PIPE
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from prettytable import PrettyTable
from utils.dataset_filter import *

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

        self.root = root            # Dataset root
        self.train = train          # Whether train mode (True) or test
        self.images = []            # List of all the images path
        self.labels = []            # List of the corresponding labels (in strings)
        self.label2positions = {}   # Dictionary which divides all the images into labels, i,e.
        # keys: string labels - values: list of ints which are the indices of the images in the self.image varaible
        # self.label2positions = {"dress": [0,
        #                                   1,
        #                                   ...],
        #                         "bakery": [423343,
        #                                    843234,
        #                                    ...]
        #                         }

        self.random_state = np.random.RandomState(29)
        self.transforms = get_training_augmentations() if train else get_validation_augmentations()
        self.code_path = os.getcwd() # path in which the code has been run

        count = 0
        # --- HANDLE PLACES DATASET ---
        if 'places' in datasets:
            # Download the dataset if not is already downloaded
            if not exists(join(self.root, "places2-mit-dataset")):
                os.system(f"bash {join(self.code_path, 'utils', 'download_placesdataset.sh')} {self.root}")
                # subprocess.run(f"bash /utils/download_placesdataset.sh {self.root}")

            images_num = sum([len(glob(join(path, "*.jpg"))) for path, _, _ in os.walk(join(self.root, "places2-mit-dataset", "train_256_places365standard", "data_256"))])
            pbar = tqdm(total=images_num, desc="Loading places2-mit-dataset")

            for subfolder in sorted(
                    glob(join(self.root, "places2-mit-dataset", "train_256_places365standard", "data_256", "*"))):
                for subsubfolder in sorted(glob(join(subfolder, "*"))):
                    if len(sorted(glob(join(subsubfolder, "*jpg")))) > 0:
                        self.images += sorted(glob(join(subsubfolder, "*jpg")))
                        self.labels += [subsubfolder.split("/")[-1]] * len(glob(join(subsubfolder, "*jpg")))

                        if subsubfolder.split("/")[-1] in self.label2positions:
                            self.label2positions[subsubfolder.split("/")[-1]] += range(count, count+len(glob(join(subsubfolder, "*jpg"))))
                        else:
                            self.label2positions[subsubfolder.split("/")[-1]] = list(range(count, count+len(glob(join(subsubfolder, "*jpg")))))

                        count += len(glob(join(subsubfolder, "*jpg")))
                        pbar.update(len(glob(join(subsubfolder, "*jpg"))))
                    else:
                        for subsubsubfolder in os.listdir(subsubfolder):
                            subsubsubfolder = join(subsubfolder, subsubsubfolder)
                            self.images += sorted(glob(join(subsubsubfolder, "*jpg")))
                            self.labels += ["_".join(subsubsubfolder.split("/")[-2:])] * len(glob(join(subsubsubfolder, "*jpg")))

                            if "_".join(subsubsubfolder.split("/")[-2:]) in self.label2positions:
                                self.label2positions["_".join(subsubsubfolder.split("/")[-2:])] += range(count, count + len(
                                    glob(join(subsubsubfolder, "*jpg"))))
                            else:
                                self.label2positions["_".join(subsubsubfolder.split("/")[-2:])] = list(
                                    range(count, count + len(glob(join(subsubsubfolder, "*jpg")))))

                            count += len(glob(join(subsubsubfolder, "*jpg")))
                            pbar.update(len(glob(join(subsubfolder, "*jpg"))))

            pbar.close()

        # --- HANDLE APPAREL DATASET ---
        if 'apparel' in datasets:
            # Download the dataset if not is already downloaded
            if not exists(join(self.root, "apparel-images-dataset")):
                os.system(f"bash {join(self.code_path, 'utils', 'download_appareldataset.sh')} {self.root}")

            images_num = sum([len(glob(join(path, "*.jpg"))) for path, _, _ in os.walk(join(self.root, "apparel-images-dataset"))])
            pbar = tqdm(total=images_num, desc="Loading apparel-images-dataset")

            for subfolder in sorted(glob(join(self.root, "apparel-images-dataset", "*"))):
                self.images += sorted(glob(join(subfolder, "*jpg")))
                self.labels += [subfolder.split("/")[-1].split("_")[-1]] * len(glob(join(subfolder, "*jpg")))

                if subfolder.split("/")[-1].split("_")[-1] in self.label2positions:
                    self.label2positions[subfolder.split("/")[-1].split("_")[-1]] += range(count, count + len(glob(join(subfolder, "*jpg"))))
                else:
                    self.label2positions[subfolder.split("/")[-1].split("_")[-1]] = list(range(count, count + len(
                        glob(join(subfolder, "*jpg")))))

                count += len(glob(join(subfolder, "*jpg")))
                pbar.update(len(glob(join(subfolder, "*jpg"))))
            pbar.close()

        # --- HANDLE ALIBABA DATASET ---
        if 'alibaba' in datasets:
            if not exists(join(self.root, "alibaba-goods-dataset")):
                os.system(f"bash {join(self.code_path, 'utils', 'download_alibabadataset.sh')} {self.root}")

            images_num = sum([len(glob(join(path, "*.jpg"))) for path, _, _ in os.walk(join(self.root, "alibaba-goods-dataset", "goods_categories"))])
            pbar = tqdm(total=images_num, desc="Loading alibaba-goods-dataset")

            for subfolder in sorted(glob(join(self.root, "alibaba-goods-dataset", "goods_categories", "*"))):
                self.images += sorted(glob(join(subfolder, "*jpg")))
                self.labels += [subfolder.split("/")[-1]] * len(glob(join(subfolder, "*jpg")))

                if subfolder.split("/")[-1] in self.label2positions:
                    self.label2positions[subfolder.split("/")[-1]] += range(count, count + len(
                        glob(join(subfolder, "*jpg"))))
                else:
                    self.label2positions[subfolder.split("/")[-1]] = list(range(count, count + len(
                        glob(join(subfolder, "*jpg")))))

                count += len(glob(join(subfolder, "*jpg")))
                pbar.update(len(glob(join(subfolder, "*jpg"))))
            pbar.close()

        # --- HANDLE ARTWORKS DATASET ---
        if 'artworks' in datasets:
            if not exists(join(self.root, "best-artworks-of-all-time")):
                os.system(f"bash {join(self.code_path, 'utils', 'download-artworksdataset.sh')} {self.root}")

            images_num = sum([len(glob(join(path, "*.jpg"))) for path, _, _ in
                              os.walk(join(self.root, "best-artworks-of-all-time", "images"))])
            pbar = tqdm(total=images_num, desc="Loading artworks-dataset")

            for subfolder in sorted(glob(join(self.root, "best-artworks-of-all-time", "images", "images", "*"))):
                self.images += sorted(glob(join(subfolder, "*jpg")))
                self.labels += [subfolder.split("/")[-1]] * len(glob(join(subfolder, "*jpg")))

                if subfolder.split("/")[-1] in self.label2positions:
                    self.label2positions[subfolder.split("/")[-1]] += range(count, count + len(
                        glob(join(subfolder, "*jpg"))))
                else:
                    self.label2positions[subfolder.split("/")[-1]] = list(range(count, count + len(
                        glob(join(subfolder, "*jpg")))))

                count += len(glob(join(subfolder, "*jpg")))
                pbar.update(len(glob(join(subfolder, "*jpg"))))
            pbar.close()

        # --- HANDLE OBJECTNET DATASET ---
        # REMINDER: train has to be false as ObjectNet dataset cannot be used for training purposes bc of its licence.
        if not train and 'objectnet' in datasets:
            if not exists(join(self.root, "objectnet")):
                os.makedirs(join(self.root, "objectnet"), exist_ok=True)
                os.system(f"bash {join(self.code_path, 'utils', 'download_objectnet.sh')} {join(self.root, 'objectnet')}")

            images_num = sum([len(glob(join(path, "*.png"))) for path, _, _ in
                              os.walk(join(self.root, "objectnet"))])
            pbar = tqdm(total=images_num, desc="Loading objectnet")

            for split_folder in sorted(glob(join(self.root, "objectnet", "*"))):
                for class_folder in sorted(glob(join(split_folder, "*", "images", "*"))):
                    self.images += sorted(glob(join(class_folder, "*")))
                    self.labels += [class_folder.split("/")[-1]] * len(glob(join(class_folder, "*")))

                    pbar.update(len(glob(join(class_folder, "*"))))
            pbar.close()

        self.labels2idx = {}
        for idx, label in enumerate(np.unique(self.labels)):
            self.labels2idx[label] = idx

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

            positive_image = self.images[np.random.choice(self.label2positions[self.labels[index]])]
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