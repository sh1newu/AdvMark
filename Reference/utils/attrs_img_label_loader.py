import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from glob import glob

class attrsImgLabelDataset(Dataset):

    def __init__(self, real_path, fake_path, image_size, attr_path="celeba"):
        super(attrsImgLabelDataset, self).__init__()
        self.image_size = image_size
        self.image_dir = real_path
        #self.real_images = sorted(glob(real_path))
        self.fake_images = sorted(glob(fake_path))
        #self.image_list = self.real_images + self.fake_images
        #self.label_list = [0] * len(self.real_images) + [1] * len(self.fake_images)
        if attr_path[0:len("celebahq")] != "celebahq":
            self.attr_path = 'network/noise_layers/stargan/list_attr_celeba.txt'
        else:
            self.attr_path = 'network/noise_layers/stargan/CelebAMask-HQ-attribute-anno.txt'

        self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        self.list = [] # os.listdir(path)]
        self.attr2idx = {}
        self.idx2attr = {}
        self.transform = transforms.Compose([
            #transforms.Resize((int(self.image_size * 1.1), int(self.image_size * 1.1))),
            #transforms.RandomCrop((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.preprocess()

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        for i, line in enumerate(lines):
            split = line.split()
            basename = os.path.basename(split[0])
            filename = os.path.splitext(basename)[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if os.path.exists(os.path.join(self.image_dir, str(filename).zfill(5) + ".png")):
                self.list.append([str(filename).zfill(5) + ".png", label])

    def __getitem__(self, index):

        if index < len(self.list):
            """Return one image and its corresponding attribute label."""
            filename, label = self.list[index]
            image = Image.open(os.path.join(self.image_dir, filename)).convert("RGB")
            return self.transform(image), torch.FloatTensor(label), 0
        else:
            image = Image.open(self.fake_images[index - len(self.list)]).convert("RGB")
            image = self.transform(image)
            return image, torch.FloatTensor([1, 1, 1, 1, 1]), 1

    def __len__(self):
        return len(self.list) + len(self.fake_images)
