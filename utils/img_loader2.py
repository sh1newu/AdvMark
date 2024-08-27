import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from glob import glob
import random

class ImgDataset2(Dataset):

    def __init__(self, real_path, fake_path, image_size, transform=None, makeblance=None):
        super(ImgDataset2, self).__init__()
        self.image_size = image_size
        #self.path = path
        #self.list = os.listdir(path)
        real_images = sorted(glob(real_path))
        fake_images = sorted(glob(fake_path))

        if makeblance:
            rate = len(fake_images) // len(real_images)
            res = len(fake_images) - rate * len(real_images)
            real_images = real_images * rate + random.sample(real_images, res)

        self.image_list = real_images + fake_images
        self.label_list = [0] * len(real_images) + [1] * len(fake_images)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
            #transforms.Resize((299, 299)),
            #transforms.RandomCrop((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert("RGB")
        image = self.transform(image)
        if image is not None:
            return image, self.label_list[index]

    def __len__(self):
        return len(self.image_list)
