import os

path = os.path.join('./data','test')
for img in os.listdir(path):
    img = img.split('.')[0]
    if len(img) != 5:
        print(img)