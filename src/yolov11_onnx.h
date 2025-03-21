#ifndef YOLOV11_ONNX_H
#define YOLOV11_ONNX_H

#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

using namespace std;

struct Detection
{
    float x;
    float y;
    float w;
    float h;
    string label;
    float confidence;
};

class Yolov11_Onnx
{
private:
    string onnx_model_path_;
    cv::Size input_shape_;
    float confidence_threshold_;
    float nms_threshold_;
    vector<string> label_list_;
    double resize_ratio_w_;
    double resize_ratio_h_;
    Ort::Env env_;
    Ort::Session session_;
    vector<float> input_data;          // Store input data
    Ort::Value input_tensor;           // Store input tensor
    vector<Ort::Value> output_tensors; // Store output tensors
    vector<Detection> obj_detection;   // Store detected objects

public:
    Yolov11_Onnx(const string &onnx_model_path,
                 const vector<string> &label_list = {"drone"},
                 const cv::Size &input_shape = cv::Size(640, 640),
                 float confidence_threshold = 0.4f,
                 float nms_threshold = 0.5f);

    ~Yolov11_Onnx() = default;

    void preprocessing(const cv::Mat &frame);
    void postprocessing(const vector<Ort::Value> &output_tensor);
    void detect(const cv::Mat &image);

    const vector<Detection> &get_detections() const { return obj_detection; }
};

#endif