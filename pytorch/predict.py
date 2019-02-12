import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

from torch import jit

import io
import time
import argparse
import cv2

from vgg import VGGNet
from utils import try_load

# Check device
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat',
'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def predict(model, image, test=False):
    # apply transform and convert BGR -> RGB
    x = image[:, :, (2, 1, 0)]
    #print('Image shape: {}'.format(x.shape))
    # H x W x C -> C x H x W for conv input
    x = torch.from_numpy(x).permute(2, 0, 1).to(device)
    torch.set_printoptions(threshold=5000)

    to_norm_tensor = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    img_tensor = to_norm_tensor(x.float().div_(255))
    #print('Image tensor: {}'.format(img_tensor))
    #print('Image tensor shape: {}'.format(img_tensor.shape))
    img_tensor.unsqueeze_(0) # add a dimension for the batch
    #print('New shape: {}'.format(img_tensor.shape))

    if test:
        ttime = 0
        for i in range (15):
            t0 = time.time()
            with torch.no_grad():
                # forward pass
                outputs = model(img_tensor)
            if use_cuda:
                torch.cuda.synchronize() # wait for operations to be complete
            tf = time.time() - t0
            ttime += tf if i > 0 else 0
            score, predicted = outputs.max(1)
            #print(outputs)
            print(f'Predicted: {classes[predicted.item()]} | {score.item()}')
            print(f'Forward pass time: {tf} seconds')
        print(f'Avg forward pass time (excluding first): {ttime/14} seconds')
    else:
        t0 = time.time()
        with torch.no_grad():
            # forward pass
            outputs = model(img_tensor)
        if use_cuda:
            torch.cuda.synchronize()
        tf = time.time() - t0
        score, predicted = outputs.max(1)
        #print(outputs)
        print(f'Predicted: {classes[predicted.item()]} | {score.item()}')
        print(f'Forward pass time: {tf} seconds')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VGGNet Predict Tool')
    parser.add_argument('mtype', type=str, choices=['pytorch', 'torch-script'], help='Model type')
    parser.add_argument('--model', type=str, default='../data/VGG16model.pth', help='Pre-trained model')
    parser.add_argument('--classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--input', type=int, default=32, help='Network input size')
    parser.add_argument('--image', type=str, default='../data/dog.png', help='Input image')
    parser.add_argument('--test_timing', type=int, default=0, help='Test timing with multiple forward pass iterations')
    args = parser.parse_args()

    # Model
    print('==> Building model...')
    if args.mtype == 'pytorch':
        model = VGGNet('D-DSM', num_classes=args.classes, input_size=args.input) # depthwise separable
        # Load model
        print('==> Loading PyTorch model...')
        model.load_state_dict(try_load(args.model))
        model.eval()
        model.to(device)
    else:
        print('==> Loading Torch Script model...')
        # Load ScriptModule from io.BytesIO object
        with open(args.model, 'rb') as f:
            buffer = io.BytesIO(f.read())
        model = torch.jit.load(buffer, map_location=device)
        #print('[WARNING] ScriptModules cannot be moved to a GPU device yet. Running strictly on CPU for now.')
        #device = torch.device('cpu') # 'to' is not supported on TracedModules (yet)

    # if device.type == 'cuda':
    #     cudnn.benchmark = True
    #     model = torch.nn.DataParallel(model)

    t0 = time.perf_counter()
    predict(model, cv2.imread(args.image), test=args.test_timing)
    print(f'Total time: {time.perf_counter()-t0} seconds')