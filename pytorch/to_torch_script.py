import torch
import argparse

from torch.jit import trace

from vgg import VGGNet
from utils import try_load

# Check device    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('[Device] {}'.format(device))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Model to Torch Script')
    parser.add_argument('mode', type=str, choices=['train', 'eval'], help='Model mode')
    parser.add_argument('--classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--input', type=int, default=32, help='Network input size')
    parser.add_argument('--model', type=str, default='../data/VGG16model.pth', help='Model to trace')
    parser.add_argument('--save', type=str, default='../data/VGG16', help='Traced model save path')
    args = parser.parse_args()

    example_input = torch.rand(1, 3, args.input, args.input)
    # TracedModule objects do not inherit the .to() or .eval() methods

    if args.mode == 'train':
        print('==> Building model...')
        model = VGGNet('D-DSM', num_classes=args.classes, input_size=args.input)
        #model.to(device)
        model.train()

        # convert to Torch Script
        print('==> Tracing model...')
        traced_model = trace(model, example_input)

        # save model for training
        traced_model.save(args.save + '-traced-train.pt')
    else:
        # load "normal" pytorch trained model
        print('==> Building model...')
        model = VGGNet('D-DSM', num_classes=args.classes, input_size=args.input)
        print('==> Loading pre-trained model...')
        model.load_state_dict(try_load(args.model))
        #model = model.to(device)
        model.eval()

        # convert to Torch Script
        print('==> Tracing model...')
        traced_model = trace(model, example_input)

        # save model for eval
        traced_model.save(args.save + '-traced-eval.pt')
