import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import config


class MyModel(nn.Module):
    def __init__(self, gpu=False):
        super(MyModel, self).__init__()
        self.gpu = gpu
        
        # input size: 4 * 50 * 130
        self.frontend_feat = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M']
        self.frontend = make_layers(self.frontend_feat,in_channels=config.CHANNELS) # 256 * 5 *7
        
        # flatten here
        self.drop = nn.Dropout(0.5)
        self.nh, self.nw = config.HEIGHT, config.WIDTH
        for i in range(5):
            self.nh += 4
            self.nw += 4
            self.nh = self.nh // 2
            self.nw = self.nw // 2
        assert self.nh > 1 and self.nw > 1, '图像尺寸过小，请放大图像后再试！'
        self.fc1 = nn.Linear(256 * self.nh * self.nw, 360)
        self.fc2 = nn.Linear(360, len(config.CHARS) * config.LEN)

        if self.gpu:
            self.to('cuda')

    def forward(self, x):
        '''x: N*4*50*130'''
        x = self.frontend(x)
        x = x.view(-1, 256 * self.nh * self.nw)  # flatten here
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).view(-1, config.LEN, len(config.CHARS))
        x = F.softmax(x, dim=2)
        x = x.view(-1, config.LEN * len(config.CHARS))
        return x
    
    def save(self, name, folder='./models'):
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, name)
        torch.save(self.state_dict(), path)

    def load(self, name, folder='./models'):
        path = os.path.join(folder, name)
        map_location = 'cpu' if self.gpu == False else 'cuda'
        static_dict = torch.load(path, map_location)
        self.load_state_dict(static_dict)
        self.eval()


def make_layers(cfg, in_channels = 3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2)# 使用padding=2准确率会更高
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == "__main__":
    model = MyModel(gpu=False)
    import torch
    x = torch.ones((1,4,50,130),dtype=torch.float)
    r = model(x)
    print(r.shape)



