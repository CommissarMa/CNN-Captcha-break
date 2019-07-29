import torch.nn as nn
import torch.nn.functional as F
import os


class MyModel(nn.Module):
    def __init__(self, gpu=False):
        super(MyModel, self).__init__()
        self.gpu = gpu
        # size: 4 * 50 * 130
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)  # 32 * 50 * 130
        self.pool1 = nn.MaxPool2d(2)  # 32 * 25 * 65
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64 * 25 * 65
        self.pool2 = nn.MaxPool2d(2)  # 64 * 12 * 32
        # flatten here
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 12 * 32, 360)
        self.fc2 = nn.Linear(360, 36 * 5)

        if self.gpu:
            self.to('cuda')

    def forward(self, x):
        '''x: N*4*50*130'''
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 64 * 12 * 32)  # flatten here
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).view(-1, 5, 36)
        x = F.softmax(x, dim=2)
        x = x.view(-1, 5 * 36)
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


if __name__ == "__main__":
    model = MyModel(gpu=True)
    import torch
    x = torch.ones((1,4,50,130),dtype=torch.float).to('cuda')
    r = model(x)
    print(r.shape)



