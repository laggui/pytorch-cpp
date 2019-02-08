import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
import time
import io

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from torch import jit

from vgg import VGGNet

# Check device    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    print_freq = 10 # print every 10 batches
    train_loss = 0
    correct = 0
    total = 0
    print('\nEpoch: %d' % epoch)
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        
        # record loss and accuracy
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % print_freq == 0:
            print('Batch: %d, Loss: %.3f | Acc: %.3f%% (%d/%d)' % (batch_idx+1, train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def validate(model, val_loader, criterion):
    model.eval()
    print_freq = 10 # print every 10 batches
    val_loss = 0.0
    
    with torch.no_grad(): # no need to track history
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # compute output
            outputs = model(inputs)        
            loss = criterion(outputs, targets)

            # record loss
            val_loss += loss.item()

            if batch_idx % print_freq == 0:
                print('Validation on Batch: %d, Loss: %f' % (batch_idx+1, val_loss/(batch_idx+1)))
    return val_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VGGNet Training Tool')
    parser.add_argument('dataset', type=str, choices=['cifar10'], help='Dataset') # only cifar10 support for now
    parser.add_argument('--upscale', type=int, default=0, help='Upscale to 224x224 for test purposes')
    parser.add_argument('--output', type=str, default='VGG16model.pth', help='Model output name')
    args = parser.parse_args()

    #cifar10 = True if args.dataset == 'cifar10' else False
    num_classes = 10
    input_size = 224 if args.upscale else 32
    # Load CIFAR10 dataset
    print('==> Preparing data...')
    transform_train = transforms.Compose([
        transforms.Resize(input_size), # for testing purposes
        transforms.RandomCrop(input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=32 if args.upscale else 128, shuffle=True, num_workers=4)

    # Model
    print('==> Building model...')
    #model = VGGNet('D', num_classes=10, input_size=32) # VGG16 is configuration D (refer to paper)
    model = VGGNet('D-DSM', num_classes=num_classes, input_size=input_size) # depthwise separable
    model = model.to(device)

    if device.type == 'cuda':
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model)

    # Training
    num_epochs = 200
    lr = 0.1
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)

    print('==> Training...')
    train_time = 0
    #scheduler = ReduceLROnPlateau(optimizer, 'min')
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1) # adjust lr by factor of 10 every 100 epochs
    for epoch in range(num_epochs):
        t0 = time.time()
        # train one epoch
        train(model, train_loader, criterion, optimizer, epoch)
        t1 = time.time() - t0
        print('{} seconds'.format(t1))
        train_time += t1

        # validate
        #val_loss = validate(model, val_loader, criterion)
        # adjust learning rate with scheduler
        #scheduler.step(val_loss)
        scheduler.step()
        
    print('==> Finished Training: {} seconds'.format(train_time))
    # Save trained model
    torch.save(model.state_dict(), args.output)
