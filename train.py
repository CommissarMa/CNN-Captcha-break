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
import numpy as np


def loss_batch(model, loss_func, data, opt=None):
    # xb: torch.Size([1, 4, 50, 130]), yb:torch.Size([1, 180])
    xb, yb = data['image'], data['label'] 
    batch_size = len(xb)
    out = model(xb) # torch.Size([1, 180])
    loss = loss_func(out, yb) # tensor(float)

    single_correct, whole_correct = 0, 0
    if opt is not None: # 训练阶段只算损失
        opt.zero_grad()
        loss.backward()
        opt.step()
    else:  # 测试阶段计算准确率
        yb = yb.view(-1, 5, 36) # 真实标签
        out_matrix = out.view(-1, 5, 36) # 预测标签
        _, ans = torch.max(yb, 2) # 真实标签的下标 torch.Size([1,5])
        _, predicted = torch.max(out_matrix, 2) # 真实标签的下标 torch.Size([1,5])
        compare = (predicted == ans) # torch.Size([1,5])，正确为1，错误为0
        single_correct = compare.sum().item() # 5个字符中预测对几个
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
        '''training phase'''
        running_loss = 0.0
        total_nums = 0
        model.train()
        for i, data in enumerate(train_dl):
            loss, _, _, s = loss_batch(model, criterion, data, opt)
            running_loss += loss * s
            total_nums += s
        ave_loss = running_loss / total_nums
        print('[Epoch {}] got training loss: {:.6f}'
                          .format(epoch + 1,  ave_loss))
        
        '''valudating phase'''
        model.eval()
        with torch.no_grad():
            losses, single, whole, batch_size = zip(
                *[loss_batch(model, criterion, data) for data in valid_dl]
            )
            total_size = np.sum(batch_size)
            val_loss = np.sum(np.multiply(losses, batch_size)) / total_size
            single_rate = 100 * np.sum(single) / (total_size * 5)
            whole_rate = 100 * np.sum(whole) / total_size
#            if single_rate > max_acc:
#                patience = 0
#                max_acc = single_rate
#                model.save('pretrained')
            
            print('After epoch {}: \n'
              '\tLoss: {:.6f}\n'
              '\tSingle Acc: {:.2f}%\n'
              '\tWhole Acc: {:.2f}%'
              .format(epoch + 1, val_loss, single_rate, whole_rate))
        
        
        
        
        
        
        
        
        
        
        