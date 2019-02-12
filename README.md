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
Forward pass time: 0.0043811798095703125 seconds
Total time: 0.0052343260031193495 seconds
```

```sh
pytorch$ python predict.py torch-script --model=../data/VGG16-traced-eval.pt --image=../data/dog.png 
==> Building model...
==> Loading Torch Script model...
Predicted: dog | 10.056212425231934
Forward pass time: 0.01126241683959961 seconds
Total time: 0.012680109008215368 seconds
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
Forward pass time: 0.005976676940917969 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.004324197769165039 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.00431060791015625 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0046079158782958984 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0043218135833740234 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.004750728607177734 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.00461125373840332 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0052700042724609375 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.004312992095947266 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.004832744598388672 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.004314422607421875 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.004302263259887695 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0047190189361572266 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.005443096160888672 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.004314899444580078 seconds
Avg forward pass time (excluding first): 0.00460256849016462 seconds
Total time: 0.0730239039985463 seconds
```

**Torch Script Model (Static)**

```sh
pytorch$ python predict.py torch-script --model=../data/VGG16model-224-traced-eval.pt --image=../data/dog-224.png --input=224
==> Building model...
==> Loading Torch Script model...
Predicted: dog | 1.722057580947876
Forward pass time: 0.014840841293334961 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0043413639068603516 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0043256282806396484 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.005699634552001953 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.004336118698120117 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.004330635070800781 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0050067901611328125 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.00433039665222168 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0043239593505859375 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.0047681331634521484 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.004338264465332031 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.004318952560424805 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.004320621490478516 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.004678487777709961 seconds
Predicted: dog | 1.722057580947876
Forward pass time: 0.004454374313354492 seconds
Avg forward pass time (excluding first): 0.004540954317365374 seconds
Total time: 0.08327161299530417 seconds
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


