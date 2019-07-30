from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch
import numpy as np
from torchvision.transforms import transforms, functional
import config


ONE_HOT = torch.eye(len(config.CHARS))

class ImageDataset(Dataset):
    def __init__(self, folder='./data', phase='train', transform=None):
        self.dir = os.path.join(folder, phase)
        self.img_list = os.listdir(self.dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        label = self.img_list[idx].split('.')[0]
        path = os.path.join(self.dir, self.img_list[idx])
        img = Image.open(path) # RGBA
#        if img.mode != 'RGB':
#            img = img.convert('RGB')
        sample = {'image': img, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Word2OneHot(object):
    def __call__(self, sample):
        labels = list()
        for c in sample['label']:
            idx = config.CHARS.index(c)
            labels.append(ONE_HOT[idx])
        sample['label'] = torch.cat(labels)
        return sample


class ImgToTensor(object):
    def __call__(self, sample):
        np_img = np.asarray(sample['image'])
        image = np_img.transpose((2, 0, 1))  # H x W x C  -->  C x H x W
        sample['image'] = torch.from_numpy(image).float()
        return sample


class Normalize(transforms.Normalize):
    def __call__(self, sample):
        tensor = sample['image']
        sample['image'] = functional.normalize(
            tensor, self.mean, self.std, self.inplace)
        return sample


class ToGPU(object):
    def __call__(self, sample):
        sample['image'] = sample['image'].to('cuda')
        sample['label'] = sample['label'].float().to('cuda')
        return sample
    
def load_data(batch_size=1, gpu=True):

    # initialize transform
    mean_ = [127.5 for i in range(config.CHANNELS)]
    std_ = [128 for i in range(config.CHANNELS)]
    chains = [Word2OneHot(),
              ImgToTensor(),
              Normalize(mean_, std_)]
    if gpu:
        chains.append(ToGPU())
    transform = transforms.Compose(chains)

    # load data
    train_ds = ImageDataset(phase='train', transform=transform)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_ds = ImageDataset(phase='test', transform=transform)
    valid_dl = DataLoader(valid_ds, batch_size=1)
    return train_dl, valid_dl