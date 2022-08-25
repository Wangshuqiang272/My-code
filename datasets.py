import glob
# import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, img_size=128, mask_size=64, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = mode
        if mode == "train":
            self.files = sorted(glob.glob("%s/paris_train_original/*.jpg" % root))
        else:
            self.files = sorted(glob.glob("%s/paris_eval_gt/*.png" % root))
        # self.files = sorted(glob.glob("%s/*.jpg" % root))
        # self.files = self.files[:-2100] if mode == "train" else self.files[-2100:]

    def apply_random_mask(self, img):
        """Randomly masks image"""
        # y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        # y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        x1 = (self.img_size - self.mask_size) // 2
        y1 = (self.img_size - self.mask_size) // 2
        x2 = x1 + self.mask_size
        y2 = y1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[0, int(y1 + 4):int(y2 - 4), int(y1 + 4):int(y2 - 4)] = 2*117.0/255.0 - 1.0
        masked_img[1, int(y1 + 4):int(y2 - 4), int(y1 + 4):int(y2 - 4)] = 2*104.0/255.0 - 1.0
        masked_img[2, int(y1 + 4):int(y2 - 4), int(y1 + 4):int(y2 - 4)] = 2*123.0/255.0 - 1.0

        return masked_img, masked_part

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[0, int(i + 4): int(i + self.mask_size - 4), int(i + 4): int(i + self.mask_size - 4)] = 2*117.0/255.0 - 1.0
        masked_img[1, int(i + 4): int(i + self.mask_size - 4), int(i + 4): int(i + self.mask_size - 4)] = 2*104.0/255.0 - 1.0
        masked_img[2, int(i + 4): int(i + self.mask_size - 4), int(i + 4): int(i + self.mask_size - 4)] = 2*123.0/255.0 - 1.0

        return masked_img, i

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)]).convert('RGB')
        img = self.transform(img)
        if self.mode == "train":
            # For training data perform random mask
            masked_img, aux = self.apply_random_mask(img)
        else:
            # For test data mask the center of the image
            masked_img, aux = self.apply_center_mask(img)

        return img, masked_img, aux

    def __len__(self):
        return len(self.files)

# Test insert for wsq 
