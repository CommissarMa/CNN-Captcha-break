from my_model import MyModel
from torchvision.transforms import functional
import numpy as np
import torch
import string

from PIL import Image


class Predictor(object):
    def __init__(self, model_path, gpu=False):
        self.net = MyModel(gpu)
        self.net.load(model_path)
        
    def identify(self, img_path):
        img = Image.open(img_path)

        # to tensor
        np_img = np.asarray(img)
        image = np_img.transpose((2, 0, 1))  # H x W x C  -->  C x H x W
        img = torch.from_numpy(image).float()

        # normalize
        img = functional.normalize(img, [127.5, 127.5, 127.5, 127.5], [128, 128, 128, 128])
        if self.net.gpu == True:  # to cpu
            img = img.to('cuda')

        with torch.no_grad():
            xb = img.unsqueeze(0)
            out = self.net(xb).squeeze(0).view(5, 36)
            _, predicted = torch.max(out, 1)
            letters = string.digits + string.ascii_lowercase
            CHARS = [c for c in letters]
            ans = [CHARS[i] for i in predicted.tolist()]
            return ans


if __name__ == '__main__':
    man = Predictor('current.pth', gpu=False)
    folder = './data/test'
    import os
    import time
    start =time.time()
    for s in os.listdir(folder):
        img_path = os.path.join(folder, s)
        print(s, man.identify(img_path))
    end = time.time()
    print(end - start)
        