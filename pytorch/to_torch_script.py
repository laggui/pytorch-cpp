import torch
import argparse

from torch.jit import trace

from vgg import VGGNet

# Check device    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('[Device] {}'.format(device))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Model to Torch Script')
    parser.add_argument('mode', type=str, choices=['train', 'eval'], help='Model mode')
    args = parser.parse_args()

    example_input = torch.rand(1, 3, 32, 32)
    # TracedModule objects do not inherit the .to() or .eval() methods

    if args.mode == 'train':
        print('==> Building model...')
        model = VGGNet('D-DSM', num_classes=10, input_size=32)
        #model.to(device)
        model.train()

        # convert to Torch Script
        print('==> Tracing model...')
        traced_model = trace(model, example_input)

        # save model for training
        traced_model.save('VGG16-traced-train.pt')
    else:
        # load "normal" pytorch trained model
        print('==> Building model...')
        model = VGGNet('D-DSM', num_classes=10, input_size=32)
        print('==> Loading pre-trained model...')
        model.load_state_dict(torch.load('VGG16model.pth', map_location=torch.device('cpu')))
        #model = model.to(device)
        model.eval()

        # convert to Torch Script
        print('==> Tracing model...')
        traced_model = trace(model, example_input)

        # save model for eval
        traced_model.save('VGG16-traced-eval.pt')
