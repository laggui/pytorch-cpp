# pytorch-cpp
In this repo I experiment with PyTorch 1.0 and their new JIT compiler, as well as their C++ API Libtorch.

Currently, the repo contains a VGG16 based network implementation in PyTorch for CIFAR-10 classification (based on my [previous experiment](https://github.com/laggui/NN_compress)), and the C++ source for inference.

## pytorch/
This subdirectory includes the network's [architecture definition](pytorch/vgg.py), the [training script](pytorch/train.py), the [test script](pytorch/test.py) on the CIFAR-10 dataset, a [prediction script](pytorch/predict.py) for inference and, most importantly, the [script to convert the model to Torch Script](pytorch/to_torch_script.py).

## libtorch/
This is where you'll find the source for the network's inference in C++. In [predict.cpp](libtorch/predict.cpp), we load the Torch Script module generated in PyTorch, read the input image and pre-process it in order to feed it to our network for inference.

## Example Usage

### PyTorch Predict

```sh
pytorch$ python predict.py pytorch --model=../data/VGG16model.pth --image=../data/dog.png
==> Building model...
==> Loading PyTorch model...
Predicted: dog | 1.722057580947876
Forward pass time: 0.005918025970458984 seconds
Total time: 0.007593393325805664 seconds
```

```sh
pytorch$ python predict.py torch-script --model=../data/VGG16-traced-eval.pt --image=../data/dog.png 
==> Building model...
==> Loading Torch Script model...
Predicted: dog | 10.056212425231934
Forward pass time: 0.010970354080200195 seconds
Total time: 0.01229238510131836 seconds
```

Predictions were done using a 1080 Ti GPU. Interestingly, the traced (static) network has slower inference time. Further investigation on a more realisitc application needs to be done, since this sample example is using CIFAR-10 images (32x32 RGB, which is a very small input size), and only predicting for one sample instead of continuously predicting in real-time.

#### Further Testing

In order to realistically test the traced (static) network versus its standard (dynamic) PyTorch model counterpart, I trained the same VGG16 network (with depthwise separable convolutions) for a single epoch, and used the saved model to predict multiple times on the same input (upscaled 224x224 image of a dog from CIFAR-10).

**Standard Model (Dynamic)**

```sh
pytorch$ python predict.py pytorch --model=../data/VGG16model-224.pth --image=../data/dog-224.png --input=224
==> Building model...
==> Loading PyTorch model...
Predicted: dog | 1.722057580947876
Forward pass time: 0.005882978439331055 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0017268657684326172 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.001695871353149414 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0016937255859375 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0017061233520507812 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0016863346099853516 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0020112991333007812 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.001771688461303711 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0017118453979492188 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0016942024230957031 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0016887187957763672 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0017006397247314453 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0016946792602539062 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.001953601837158203 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0017178058624267578 seconds
Avg forward pass time (excluding first): 0.001746671540396554 seconds
Total time: 0.07191038131713867 seconds
```

**Torch Script Model (Static)**

```sh
pytorch$ python predict.py torch-script --model=../data/VGG16model-224-traced-eval.pt --image=../data/dog-224.png --input=224
==> Building model...
==> Loading Torch Script model...
Device: cuda
Predicted: dog | 1.722057580947876
Forward pass time: 0.013720512390136719 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.001064300537109375 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0010313987731933594 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0010249614715576172 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.001073598861694336 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0010318756103515625 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0010318756103515625 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0010263919830322266 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.001024007797241211 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0010232925415039062 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0010209083557128906 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0010225772857666016 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.001026153564453125 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0010263919830322266 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0010259151458740234 seconds
Avg forward pass time (excluding first): 0.0010324035372052873 seconds
Total time: 0.07527804374694824 seconds
```

As we can see, the results are what we expected (1.7466ms vs 1.0324ms average), as opposed to the previous inference time we got on a 32x32 image for a single forward pass. In both cases, the first forward pass takes longer (5.891 ms vs 13.796 ms) than the following. But, for a static input (same dimensions), the Torch Sript model is faster on the following forward passes, which is much more representative of real use-cases.

### Libtorch
Before running our prediction, we need to compile the source. In your `libtorch` directory, create a build directory and compile+build the application from source.

```sh
libtorch$ mkdir build 
libtorch$ cd build
libtorch/build$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
-- The C compiler identification is GNU 5.4.0
-- The CXX compiler identification is GNU 5.4.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
.
.
.
-- Configuring done
-- Generating done
-- Build files have been written to: libtorch/build
libtorch/build$ make
Scanning dependencies of target vgg-predict
[ 50%] Building CXX object CMakeFiles/vgg-predict.dir/predict.cpp.o
[100%] Linking CXX executable vgg-predict
[100%] Built target vgg-predict  
```

You're now ready to run the application.

```sh
libtorch/build$ ./vgg-predict ../../data/VGG16model.pth ../../data/dog.png
Model loaded
Moving model to GPU
Predicted: dog | 10.0562
Time: 0.009481 seconds
```

### TO-DO

- Update libtorch C++ example


