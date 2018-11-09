import torch
import torch.nn as nn

class VGGNet(nn.Module):
    """
    Base VGG model
    """
    def __init__(self, vgg_cfg,  num_classes=1000, input_size=224):
        super(VGGNet, self).__init__()
        self.features = self._make_layers(vgg_cfg)
        self.classifier = nn.Sequential(
                                        #nn.Linear(int((input_size/(2**5))*(input_size/(2**5))*512), 4096),
                                        #nn.ReLU(inplace=True),
                                        #nn.Dropout(), # Dropout of 0.5 is default, as in paper
                                        #nn.Linear(4096, 4096),
                                        #nn.ReLU(inplace=True),
                                        #nn.Dropout(),
                                        #nn.Linear(4096, num_classes) # For input_size = 224
                                        nn.Linear(int((input_size/(2**5))*(input_size/(2**5))*512), num_classes) # For input_size = 32 (CIFAR-10)
                        		        )

    def forward(self, x): # computation performed at every call
        x = self.features(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.classifier(x)
        return x
    
    def _make_layers(self, vgg_cfg):
        """
        For portability, if other configurations of the network
        need to be defined (e.g., A, B, C or E from the VGG paper)
        
        D: Standard D config as per the VGG paper
        D-DSM: Compact D config with depthwise separable (DS) convolutions with maxpool layers (as per the classic VGGNet)
        D-DS: Compact D config with depthwise separable (DS) convolutions, using conv stride=2 instead of maxpool layers to reduce spatial size
        """
        cfg = {
		'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
		'D-DSM': [64, (64, 1), 'M', (128, 1), (128, 1), 'M', (256, 1), (256, 1), (256, 1), 'M', (512, 1), (512, 1), (512, 1), 'M', (512, 1), (512, 1), (512, 1), 'M'],
		'D-DS': [64, (64, 2), (128, 2), (128, 1), (256, 2), (256, 1), (256, 1), (512, 2), (512, 1), (512, 1), (512, 2), (512, 1), (512, 1)]
		}
        
        in_channels = 3 # RGB images
        layers = []
        
        for x in cfg[vgg_cfg]:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if isinstance(x, int):
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                    in_channels = x # Next input is the size of current output
                else: # Depthwise separable
                    layers += [nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=x[1], padding=1, groups=in_channels, bias=True),
                               nn.BatchNorm2d(in_channels),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(in_channels, x[0], kernel_size=1, padding=0, bias=True),
                               nn.BatchNorm2d(x[0]),
                               nn.ReLU(inplace=True)]
                    in_channels = x[0] # Next input is the size of current output
        
        return nn.Sequential(*layers)