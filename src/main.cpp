#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolov11_onnx.h"
#include "bbox.h"
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

int main()
{
    const string image_path = "/home/danz/Downloads/Drone_detect/Data_bla/DroneTestDataset/Drone_TestSet/VS_P7637.jpg";
    const string model_path = "/home/danz/Downloads/Drone_detect/yolov11_cpp_project/best.onnx";
    cv::Mat image_draw = cv::imread(image_path);

    Yolov11_Onnx detector(model_path);

    detector.detect(image_draw);

    cv::Mat result_frame = Bbox::draw_box(image_draw, detector.get_detections());

    cv::imshow("Detection result", result_frame);
    cv::waitKey(0);
    cv::destroyAllWindows();

    /************************Measurement accuracy with ~ 10^4 images************************/
    // const string output_txt_path = "/home/danz/Downloads/Drone_detect/Data_bla/DroneTestDataset/infer_detections_cpp.txt";
    // const string image_folder = "/home/danz/Downloads/Drone_detect/Data_bla/DroneTestDataset/Drone_TestSet";

    // ofstream output_file(output_txt_path);
    // if (!output_file.is_open())
    // {
    //     cerr << "Failed to open output file" << endl;
    //     return -1;
    // }

    // for (const auto &entry : fs::directory_iterator(image_folder))
    // {
    //     if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")
    //     {
    //         string image_path = entry.path().string();
    //         vector<Detection> detections = detector.detect(image_path);

    //         if (!detections.empty())
    //         {
    //             for (const auto &detection : detections)
    //             {
    //                 float x_center = detection.x;
    //                 float y_center = detection.y;
    //                 float x_min = x_center - detection.w / 2;
    //                 float y_min = y_center - detection.h / 2;
    //                 float x_max = x_center + detection.w / 2;
    //                 float y_max = y_center + detection.h / 2;

    //                 output_file << entry.path().filename().string() << " "
    //                             << static_cast<int>(x_min) << " "
    //                             << static_cast<int>(y_min) << " "
    //                             << static_cast<int>(x_max) << " "
    //                             << static_cast<int>(y_max) << " "
    //                             << "drone" << " "
    //                             << fixed << setprecision(2) << detection.confidence << endl;
    //             }
    //         }

    //         cv::Mat result_frame = Bbox::draw_box(image_path, detections);

    //         cv::imshow("Detection Result", result_frame);
    //         cv::waitKey(1);
    //     }
    // }

    // output_file.close();
    // cout << "Results have been saved to " << output_txt_path << endl;
    /************************************End tune*********************************************/

    return 0;
}