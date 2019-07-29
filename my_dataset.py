from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
from PIL import Image
import string
import torch
import numpy as np
from torchvision.transforms import transforms, functional

letters = string.digits + string.ascii_lowercase
CHARS = [c for c in letters]

ONE_HOT = torch.eye(len(CHARS))

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
            idx = CHARS.index(c)
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
    chains = [Word2OneHot(),
              ImgToTensor(),
              Normalize([127.5, 127.5, 127.5, 127.5], [128, 128, 128, 128])]
    if gpu:
        chains.append(ToGPU())
    transform = transforms.Compose(chains)

    # load data
    train_ds = ImageDataset(phase='train', transform=transform)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_ds = ImageDataset(phase='test', transform=transform)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size)
    return train_dl, valid_dl

if __name__ == '__main__':
    td, vd = load_data(batch_size=1, gpu=True)
    for i,data in enumerate(td):
        img = data['image']
        label = data['label']
        if label.shape != torch.Size([1, 180]):
            print(img.shape)
            print(label.shape)
#        print(img.shape)
#        print(label)
#        if img.shape[]
#        print(i)
#        break