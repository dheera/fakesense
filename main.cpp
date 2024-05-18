#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <cstdlib>

using namespace cv;
using namespace std;

void create_virtual_camera() {
    struct stat buffer;
    if (stat("/dev/video1", &buffer) != 0) {
        system("sudo modprobe v4l2loopback video_nr=1");
    }
}

Mat estimate_depth(torch::jit::script::Module& model, const Mat& image) {
    Mat input;
    cvtColor(image, input, COLOR_BGR2RGB);
    input.convertTo(input, CV_32F, 1.0 / 255);

    torch::Tensor tensor_image = torch::from_blob(input.data, {1, input.rows, input.cols, 3}, torch::kFloat32);
    tensor_image = tensor_image.permute({0, 3, 1, 2});
    tensor_image = torch::upsample_bicubic2d(tensor_image, {384, 384}, /*align_corners=*/false);

    torch::Tensor depth_map = model.forward({tensor_image}).toTensor();
    depth_map = torch::upsample_bicubic2d(depth_map.unsqueeze(1), {input.rows, input.cols}, /*align_corners=*/false).squeeze().detach();

    Mat depth_image(input.rows, input.cols, CV_32F, depth_map.data_ptr());
    normalize(depth_image, depth_image, 0, 1, NORM_MINMAX);
    return depth_image;
}

int main() {
    create_virtual_camera();

    torch::jit::script::Module model;
    try {
        model = torch::jit::load("midas.torchscript.pt");
    } catch (const c10::Error& e) {
        cerr << "Error loading the model\n";
        return -1;
    }

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open webcam\n";
        return -1;
    }

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        Mat depth_map = estimate_depth(model, frame);
        depth_map.convertTo(depth_map, CV_8U, 255);

        imshow("Depth Map", depth_map);
        if (waitKey(1) == 'q') {
            break;
        }

        stringstream ss;
        ss << "ffmpeg -y -f rawvideo -pixel_format gray -video_size " << depth_map.cols << "x" << depth_map.rows << " -i - -f v4l2 /dev/video1";
        string command = ss.str();
        FILE* ffmpeg = popen(command.c_str(), "w");
        fwrite(depth_map.data, 1, depth_map.total(), ffmpeg);
        pclose(ffmpeg);
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

