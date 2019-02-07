import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

from torch import jit
from PIL import Image

import io
import time
import argparse
import cv2

from vgg import VGGNet

# Check device    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat',
'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def predict(model, image):
    # apply transform and convert BGR -> RGB
    x = image[:, :, (2, 1, 0)]
    #print('Image shape: {}'.format(x.shape))
    # H x W x C -> C x H x W for conv input
    x = torch.from_numpy(x).permute(2, 0, 1)
    torch.set_printoptions(threshold=5000)

    to_norm_tensor = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    img_tensor = to_norm_tensor(x.float().div_(255))
    #print('Image tensor: {}'.format(img_tensor))
    #print('Image tensor shape: {}'.format(img_tensor.shape))
    img_tensor.unsqueeze_(0).to(device) # add a dimension for the batch
    #print('New shape: {}'.format(img_tensor.shape))

    with torch.no_grad():
        # forward pass
        outputs = model(img_tensor)
    score, predicted = outputs.max(1)
    #print(outputs)
    print('Predicted: {} | {}'.format(classes[predicted.item()], score.item()))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VGGNet Predict Tool')
    parser.add_argument('mtype', type=str, choices=['pytorch', 'torch-script'], help='Model type')
    parser.add_argument('--model', type=str, default='../data/VGG16model.pth', help='Pre-trained model')
    parser.add_argument('--image', type=str, default='../data/dog.png', help='Input image')
    args = parser.parse_args()

    # Model
    print('==> Building model...')
    if args.mtype == 'pytorch':
        model = VGGNet('D-DSM', num_classes=10, input_size=32) # depthwise separable
        # Load model
        print('==> Loading PyTorch model...')
        model.load_state_dict(torch.load(args.model))
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

    if device.type == 'cuda:0':
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model)

    t0 = time.time()
    predict(model, cv2.imread(args.image))
    print('Time: {} seconds'.format(time.time()-t0))