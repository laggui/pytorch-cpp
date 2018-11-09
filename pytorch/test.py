import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch import jit

import io
import time
import argparse

from vgg import VGGNet

# Check device    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu') # 'to' is not supported on TracedModules, ref: https://github.com/pytorch/pytorch/issues/6008

def test(model, test_loader):
    #model.eval()
    print_freq = 10 # print every 10 batches
    correct = 0
    total = 0
    
    with torch.no_grad(): # no need to track history
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # compute output
            outputs = model(inputs)        

            # record prediction accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % print_freq == 0:
                print('Batch: %d, Acc: %.3f%% (%d/%d)' % (batch_idx+1, 100.*correct/total, correct, total))
    return correct, total

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VGGNet Test Tool')
    parser.add_argument('mtype', type=str, choices=['pytorch', 'torch-script'], help='Model type')
    args = parser.parse_args()

    # Model
    print('==> Building model...')
    if args.mtype == 'pytorch':
        model = VGGNet('D-DSM', num_classes=10, input_size=32) # depthwise separable
        # Load model
        print('==> Loading PyTorch model...')
        model.load_state_dict(torch.load('VGG16model.pth'))
        model.to(device)
    else:
        print('==> Loading Torch Script model...')
        # Load ScriptModule from io.BytesIO object
        with open('VGG16-traced-eval.pt', 'rb') as f:
            buffer = io.BytesIO(f.read())
        model = torch.jit.load(buffer)
        print('[WARNING] ScriptModules cannot be moved to a GPU device yet. Running strictly on CPU for now.')
        device = torch.device('cpu') # 'to' is not supported on TracedModules (yet)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    if device.type == 'cuda':
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model)

    t0 = time.time()
    correct, total = test(model, test_loader)
    t1 = time.time()
    print('Accuracy of the network on test dataset: %f (%d/%d)' % (100.*correct/total, correct, total))
    print('Elapsed time: {} seconds'.format(t1-t0))