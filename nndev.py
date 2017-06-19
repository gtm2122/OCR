import matplotlib.pyplot as plt

import  torch.utils.data as data_utils
import torch
import torch.nn as nn
from torchvision.models import inception
from torchvision.models import Inception3
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets,models,transforms
import torch.optim as optim
import copy

import skimage.io as io
from scipy.misc import imsave
from skimage import img_as_uint
import errno 
import numpy as np
import random
import gc


import torch
import numpy as np
from torch.autograd import Variable

import sys
import os
import scipy
import cv2
from PIL import Image

### most of the code was taken from pytorch's trasnfer learning tutorial

def imshow(inp,title=None):
    # Converts image from CxWxH to WxHxC , reverses normalization and displays image
    
    inp=inp.numpy().transpose((1,2,0))
    mean=np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    inp = std*inp + mean
    plt.imshow(inp),plt.show()
    plt.pause(0.01)


def lr_scheduler(optimizer,epoch,init_lr = 0.001,lr_decay_epoch = 7):
    
    ### This is to adjust the learning rate.
    
    lr = init_lr*(0.1**(epoch//lr_decay_epoch))
    if epoch%lr_decay_epoch ==0:
        print('lr = ',lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def result(val_path,t_model,n='Res'):
    correct= 0
    # this function saves the images that were misclassifed 
    try:
        os.makedirs('/data/gabriel/OCR/OCR_data/misclas/'+str(n))
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir('/data/gabriel/OCR/OCR_data/misclas/'):
            pass
        else:
            raise
    dsets,dset_loaders,dset_sizes = load_data(val_path,10)
    
    val_load = dset_loaders['val']
    val_size = dset_sizes['val']
    val = dsets['val']
    
    t_model = t_model.cuda()
    wrong=0
    count=0
    for data in val_load:
        inp,label = data
        inputs,label = Variable(inp.cuda()),Variable(label.cuda())
        
        count+=1
       
        if(inputs.size(0)<10 and  n == 'Inception'):
            flag=1
            temp = Variable(torch.zeros((10,3,300,300)).cuda())
                
            temp[0:inputs.size(0)]=inputs
                
            inputs = temp
                    
            temp2 = Variable(torch.LongTensor((10)).cuda())
            temp2[0:labels.size(0)] = labels
            temp2[labels.size(0):] = 0
                    
            labels = temp2
        out = t_model(inputs)
        _,pred = torch.max(out.data,1)
        
        l = label.data.cpu().numpy().reshape(10)
        p = pred.cpu().numpy().T.reshape(10)
        
        eq = np.where(~(l==p))[0]
        #print(l)
        #print(p)
        #print(eq)
        if len(eq)>0:
            #print(eq)
            for i in eq:
                #print(i)
                #print(inp.numpy().shape)
                c=inp[i,:,:,:].numpy().transpose((1,2,0))
                mean=np.array([0.485,0.456,0.406])
                std = np.array([0.229,0.224,0.225])
                c= (std*c + mean)#.astype(np.uint32)
                #plt.imshow(inp),plt.show()
                #plt.pause(0.01)
                
                plt.imshow(c),plt.show()
                
                print(p[i])
                print(l[i])
                
                imsave('/data/gabriel/OCR/OCR_data/misclas/'+str(n)+'/'+str(count)+'_'+str(p[i])+'.png',c)
            correct+=10-len(eq)
            wrong+=len(eq)
    print(correct)
    print(wrong)
    print(correct/(correct+wrong))
    print(wrong/(correct+wrong))

def load_data(path,b_size=10):
    
    ### loads data as an iterator that loads batches of images as Tensors
    data_transforms = {'train':transforms.Compose([transforms.Scale(300),
                                            transforms.CenterCrop(300),
                                            #transforms.RandomCrop(300),
                                               #transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                   ,
                   'val':transforms.Compose([
                transforms.Scale(300),
                transforms.CenterCrop(300),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    }
    
    dsets = {x:datasets.ImageFolder(path+x,data_transforms[x]) for x in ['train','val']}
    dset_loaders ={x: torch.utils.data.DataLoader(dsets[x],batch_size=b_size,shuffle=True,num_workers=4)
              for x in ['train','val']}

    dset_sizes = {x:len(dsets[x]) for x in ['train','val']}
    
    return dsets,dset_loaders,dset_sizes


### Outputs the model with best validation result and it's corresponding confusion matrix
### Also prints the best epoch and corresponding best validation accuracy
def train_model(model,optimizer,criterion,lr_scheduler,data_path,batch_size, n_epochs=30,n_gpu=0,n=None):
    
    
    
    
    # 'n' parameter stands for name of model, most models from pytorch's model zoo does not require this parameter
    # this is required because models like inception have a 2 channel output
    
    
    
    dsets,dset_loaders,dset_sizes = load_data(data_path,batch_size)
    
    
    best_model=model
    best_acc = 0.0
    torch.cuda.set_device(n_gpu)
    best_epoch = 0
    model=model.cuda()
    for epoch in range(0,int(n_epochs)):
        print('Epoch = ',epoch)
               
        for phase in ['train','val']:
            if(phase == 'train'):
                model.train(True)
                
                optimizer = lr_scheduler(optimizer,epoch)
            else:
                model.train(False)
            
            c_mat = np.zeros((15,15)).astype('int') 
            running_loss = 0.0
            running_corrects = 0.0
            running_tp = 0.0
            for data in dset_loaders[phase]:
                inputs,labels = data
                
                inputs,labels = Variable(inputs.cuda()),Variable(labels.cuda())
                optimizer.zero_grad()
                
                flag=0
                if(inputs.size(0)<batch_size and  n == 'Inception'):
                    flag=1
                    temp = Variable(torch.zeros((batch_size,3,300,300)).cuda())
                    
                    temp[0:inputs.size(0)]=inputs
                    
                    inputs = temp
                    
                    temp2 = Variable(torch.LongTensor((10)).cuda())
                    
                    temp2[0:labels.size(0)] = labels
                    temp2[labels.size(0):] = 0
                    
                    labels = temp2
                    
                outputs = model(inputs)
                
                if(n=='Inception'):
#                   
                    if phase=='val':
                        
                        _,preds = torch.max(outputs.data,1)
                        loss = criterion(outputs,labels)
                    else:
                   
                        _,preds = torch.max(outputs[0].data,1)
                        loss = criterion(outputs[0],labels)
                
                else:
                    _,preds = torch.max(outputs.data,1)
                    loss = criterion(outputs,labels)
                
                
                
                if phase=='train':
                    loss.backward()
                    optimizer.step()
                
                running_loss+=loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                
                
                for i in range(0,labels.data.cpu().numpy().shape[0]):
                    
                    c_mat[labels.data.cpu().numpy()[i],preds.cpu().numpy()[i]]+=1
            
            
            epoch_loss = running_loss/dset_sizes[phase]
            epoch_acc = running_corrects/dset_sizes[phase]
            epoch_tpr = running_tp/dset_sizes[phase]
            print(phase + '\n')
            print(phase + '{} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))
            
           
            print(c_mat)
            if phase=='val' and epoch_acc>best_acc:
                best_acc=epoch_acc
                best_model=copy.deepcopy(model)
                best_epoch=epoch
                best_c = c_mat
        
        #print()
        
    print(best_acc)
    print(best_epoch)
    return best_model.cpu(),best_c        