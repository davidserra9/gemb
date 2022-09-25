"""
[1]: https://www.kaggle.com/datasets/nickj26/places2-mit-dataset
[2]: https://www.kaggle.com/datasets/trolukovich/apparel-images-dataset
"""
import os
from glob import glob
from os.path import exists, join

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class TripletGUIE(Dataset):

    def __init__(self, root, train, places, apparel):
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

        # --- HANDLE PLACES DATASET ---
        if places:
            # Download the dataset if not is already downloaded
            if not exists(join(self.root, "places2-mit-dataset")):
                os.system(f"bash /utils/download_placesdataset.sh {self.root}")

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
        if apparel:
            # Download the dataset if not is already downloaded
            if not exists(join(self.root, "apparel-images-dataset")):
                os.system(f"bash /utils/download_appareldataset.sh {self.root}")

            for subfolder in sorted(glob(join(self.root, "apparel-images-dataset", "*"))):
                crossover = int(0.8 * len(sorted(glob(join(subfolder, "*")))))
                if train:
                    self.images += sorted(glob(join(subfolder, "*")))[:crossover]
                    self.labels += [subfolder.split("/")[-1].split("_")[-1]] * crossover

                else:
                    self.images += sorted(glob(join(subfolder, "*")))[crossover:]
                    self.labels += [subfolder.split("/")[-1].split("_")[-1]] * (
                            len(sorted(glob(join(subfolder, "*")))) - crossover)

        self.label2positions = {label: np.where(np.asarray(self.labels) == label)[0]
                                for label in self.labels}

        if not self.train:
            self.triplets = [[i,
                              self.random_state.choice(self.label2positions[self.labels[i]]),
                              self.random_state.choice(self.label2positions[
                                                           np.random.choice(
                                                               list(set(self.labels) - {self.labels[i]})
                                                           )
                                                       ])
                              ]
                             for i in range(len(self.images))]

    def __getitem__(self, index):
        if self.train:
            anchor_image, anchor_label = self.images[index], self.labels[index]

            positive_image = self.images[np.random.choice(self.label2positions[anchor_label])]
            positive_label = anchor_label
            negative_label = np.random.choice(self.label2positions[np.random.choice(list(set(self.labels) - {anchor_label}))])
            negative_image = self.images[negative_label]

            anchor_image = Image.open(anchor_image)
            positive_image = Image.open(positive_image)
            negative_image = Image.open(negative_image)

        else:
            anchor_label = self.triplets[index][0]
            anchor_image = Image.open(self.images[anchor_label])

            positive_label = self.triplets[index][0]
            positive_image = Image.open(self.images[positive_label])

            negative_label = self.triplets[index][0]
            negative_image = Image.open(self.images[negative_label])

        # TODO: training and test tranformations. Use albumentations

        return (anchor_image, positive_image, negative_image), (anchor_label, positive_label, negative_label)

    def __len__(self):
        return self.n_samples  # if you want to subsample for speed


if __name__ == "__main__":
    dataset = TripletGUIE(root="/home/david/Workspace/gemb/data",
                          train=True,
                          places=True,
                          apparel=True)

    (img1, img2, img3), (l1, l2, l3) = dataset[0]

    plt.subplot(131)
    plt.title(l1)
    plt.imshow(img1)
    plt.subplot(132)
    plt.title(l2)
    plt.imshow(img2)
    plt.subplot(133)
    plt.title(l3)
    plt.imshow(img3)
    plt.show()