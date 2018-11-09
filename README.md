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
Predicted: dog | 10.056212425231934
Time: 0.06844925880432129 seconds
```

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



