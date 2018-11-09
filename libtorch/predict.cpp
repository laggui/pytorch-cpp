#include <torch/torch.h>
#include <torch/script.h> // One-stop header

#include <ctime>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> // opencv input/output
#include <opencv2/imgproc/imgproc.hpp> // cvtColor

// CIFAR-10 classes
const std::vector<std::string> classes{"plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

at::Tensor imageToTensor(cv::Mat & image);
void predict(std::shared_ptr<torch::jit::script::Module> model, cv::Mat & image);

// Adapted from https://github.com/goldsborough/examples/blob/cpp/cpp/mnist/mnist.cpp#L106
// Parameters: means and stddevs lists size must match number of channels for input Tensor
//             e.g., means and stddevs must be of size C for Tensor of shape 1 x C x H x W
struct Normalize : public torch::data::transforms::TensorTransform<> {
    Normalize(const std::initializer_list<float> & means, const std::initializer_list<float> & stddevs)
        : means_(insertValues(means)), stddevs_(insertValues(stddevs)) {}
    std::list<torch::Tensor> insertValues(const std::initializer_list<float> & values) {
        std::list<torch::Tensor> tensorList;
        for (auto val : values) {
            tensorList.push_back(torch::tensor(val));
        }
        return tensorList;
    }
    torch::Tensor operator()(torch::Tensor input) {
      std::list<torch::Tensor>::iterator meanIter = means_.begin();
      std::list<torch::Tensor>::iterator stddevIter = stddevs_.begin();
      //  Substract each channel's mean and divide by stddev in place
      for (int i{0}; meanIter != means_.end() && stddevIter != stddevs_.end(); ++i, ++meanIter, ++stddevIter){
          //std::cout << "Mean: " << *meanIter << " Stddev: " << *stddevIter << std::endl;
          //std::cout << input[0][i] << std::endl;
          input[0][i].sub_(*meanIter).div_(*stddevIter);
      }
    return input;
    }

    std::list<torch::Tensor> means_, stddevs_;
};

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "usage: vgg-predict <path-to-exported-script-module> <path-to-input-image>" << std::endl;
        return -1;
    }
    // Deserialize the ScriptModule from a file using torch::jit::load().
    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);

    assert(module != nullptr);
    std::cout << "Model loaded" << std::endl;

    // Read the image file
    cv::Mat image;
    image = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);

    // Check for invalid input
    if(! image.data ) {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    // Check for cuda
    if (torch::cuda::is_available()) {
        std::cout << "Moving model to GPU" << std::endl;
        module->to(at::kCUDA);
    }
    std::clock_t start{std::clock()};
    predict(module, image);
    std::cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " seconds" << std::endl;

    return 0;
}

at::Tensor imageToTensor(cv::Mat & image) {
    // BGR to RGB, which is what our network was trained on
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    // Split bgr interleaved channels
    cv::Mat rgb[3];
    cv::split(image, rgb);
    // Concatenate channels
    cv::Mat rgbConcat;
    cv::vconcat(rgb, 3, rgbConcat);
    
    // Convert Mat image to tensor 1 x C x H x W
    at::Tensor tensor_image = torch::from_blob(rgbConcat.data, {1, image.channels(), image.rows, image.cols}, at::kByte);
    
    // Normalize tensor values from [0, 255] to [0, 1]
    tensor_image = tensor_image.toType(at::kFloat);
    tensor_image = tensor_image.div_(255);
    return tensor_image; // 1 x C x H x W
}

void predict(std::shared_ptr<torch::jit::script::Module> module, cv::Mat & image) {
    at::Tensor tensor_image{imageToTensor(image)};

    // Normalize
    struct Normalize normalizeChannels({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010});
    tensor_image = normalizeChannels(tensor_image);
    //std::cout << "Image tensor shape: " << tensor_image.sizes() << std::endl;

    // Move tensor to CUDA memory
    tensor_image = tensor_image.to(at::kCUDA);
    // Forward pass
    at::Tensor result = module->forward({tensor_image}).toTensor();
    auto max_result = result.max(1);
    auto max_index = std::get<1>(max_result).item<float>();
    auto max_out = std::get<0>(max_result).item<float>();
    std::cout << "Predicted: " << classes[max_index] << " | " << max_out << std::endl;
}