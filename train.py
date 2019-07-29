# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:23:58 2019

@author: Ma Zhenwei
"""
from my_dataset import load_data
from my_model import MyModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def loss_batch(model, loss_func, data, opt=None):
    xb, yb = data['image'], data['label']
    batch_size = len(xb)
    out = model(xb)
    loss = loss_func(out, yb)

    single_correct, whole_correct = 0, 0
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    else:  # calc accuracy
        yb = yb.view(-1, 5, 36)
        out_matrix = out.view(-1, 5, 36)
        _, ans = torch.max(yb, 2)
        _, predicted = torch.max(out_matrix, 2)
        compare = (predicted == ans)
        single_correct = compare.sum().item()
        for i in range(batch_size):
            if compare[i].sum().item() == 5:
                whole_correct += 1
        del out_matrix
    loss_item = loss.item()
    del out
    del loss
    return loss_item, single_correct, whole_correct, batch_size


if __name__ == '__main__':
    # 是否使用GPU来训练
    use_gpu = torch.cuda.is_available()
    batch_size = 4
    epochs = 50
    
    train_dl, valid_dl = load_data(batch_size=4, gpu=use_gpu)
    model = MyModel(gpu=use_gpu)
    opt = optim.Adadelta(model.parameters())
    criterion = nn.BCELoss()  # loss function
    
    max_acc = 0
    for epoch in range(epochs):
        running_loss = 0.0
        total_nums = 0
        model.train()  # train mode
        for i, data in enumerate(train_dl):
            loss, _, _, s = loss_batch(model, criterion, data, opt)
            running_loss += loss * s
            total_nums += s
        ave_loss = running_loss / total_nums
        print('[Epoch {}][Ba got training loss: {:.6f}'
                          .format(epoch + 1,  ave_loss))
        
        
        
        
        
        
        
        
        
        
        
        
        