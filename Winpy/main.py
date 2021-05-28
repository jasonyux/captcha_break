import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image

from captcha.image import ImageCaptcha
from tqdm import tqdm
import random
import numpy as np
from collections import OrderedDict

import string

from CaptchaDataset import CaptchaDataset
from Model import Model
from Utilss import train
from Utilss import valid

model = None
characters = None
image = None
train_loader = None
valid_loader = None
dataset = None
target = None

def init():
    global model, characters, image, target, train_loader, valid_loader, dataset
    characters = string.digits + string.ascii_uppercase + string.ascii_lowercase
    # width, height, n_len, n_classes = 192, 64, 4, len(characters)#192 64
    width, height, n_len, n_classes = 150, 60, 4, len(characters)
    n_input_length = 9
    print(characters, width, height, n_len, n_classes)


    dataset = CaptchaDataset(characters, 1, width, height, n_input_length, n_len)
    image, target, input_length, label_length = dataset[0]
    print(''.join([characters[x] for x in target]), input_length, label_length)
    to_pil_image(image)


    batch_size = 70
    train_set = CaptchaDataset(characters, 1000 * batch_size, width, height, n_input_length, n_len)
    valid_set = CaptchaDataset(characters, 100 * batch_size, width, height, n_input_length, n_len)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=2)



    model = Model(n_classes, input_shape=(3, height, width))
    inputs = torch.zeros((32, 3, height, width))
    outputs = model(inputs)
    print(outputs.shape)

    model = Model(n_classes, input_shape=(3, height, width))
    model = model.cuda()
    print(model)

def decode(sequence):
    a = ''.join([characters[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != characters[0] and x != a[j+1]])
    if len(s) == 0:
        return ''
    if a[-1] != characters[0] and s[-1] != a[-1]:
        s += a[-1]
    return s

def decode_target(sequence):
    return ''.join([characters[x] for x in sequence]).replace(' ', '')

def calc_acc(target, output):
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    target = target.cpu().numpy()
    output_argmax = output_argmax.cpu().numpy()
    a = np.array([decode_target(true) == decode(pred) for true, pred in zip(target, output_argmax)])
    return a.mean()



if __name__=='__main__':
    init()

    optimizer = torch.optim.Adam(model.parameters(), 1e-3, amsgrad=True)
    epochs = 6
    for epoch in range(1, epochs + 1):
        train(model, optimizer, epoch, train_loader,characters)
        valid(model, optimizer, epoch, valid_loader,characters)

    optimizer = torch.optim.Adam(model.parameters(), 1e-4, amsgrad=True)
    epochs = 3
    for epoch in range(1, epochs + 1):
        train(model, optimizer, epoch, train_loader,characters)
        valid(model, optimizer, epoch, valid_loader,characters)
    
    
    model.eval()
    output = model(image.unsqueeze(0).cuda())
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    do = True
    while do or decode_target(target) == decode(output_argmax[0]):
        do = False
        image, target, input_length, label_length = dataset[0]
        print('true:', decode_target(target))

        output = model(image.unsqueeze(0).cuda())
        output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
        print('pred:', decode(output_argmax[0]))
        break #don't know what this is for. It got me into infinite loop
    to_pil_image(image)


    torch.save(model, 'ctc3_test.pth')
    