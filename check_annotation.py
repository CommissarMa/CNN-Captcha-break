import os


phase = 'train' # 可选： train 或者 test
path = os.path.join('./data',phase)
print('以下图像标注存在问题：')
for img in os.listdir(path):
    img = img.split('.')[0]
    if len(img) != 5:
        print(img)