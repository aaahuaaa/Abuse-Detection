#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <deque>
#include <chrono>
#include <iostream>
#include <cmath>
#include <string>

class YOLOv8_ONNX_Inference {
private:
    Ort::Env env;
    Ort::Session session;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    cv::Size input_size{640, 640};

public:
    YOLOv8_ONNX_Inference(const std::string& onnx_path) :
        env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8_ONNX")
    {
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = Ort::Session(env, onnx_path.c_str(), session_options);

        // 获取输入输出信息
        input_names.push_back(session.GetInputName(0, Ort::AllocatorWithDefaultOptions()));
        output_names.push_back(session.GetOutputName(0, Ort::AllocatorWithDefaultOptions()));
    }

    // 预处理（BGR转RGB + 尺寸调整 + 归一化）
    cv::Mat preprocess(const cv::Mat& frame) {
        cv::Mat resized, rgb;
        cv::resize(frame, resized, input_size);
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        cv::Mat float_img;
        rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);
        return float_img;
    }

    // 推理和后处理
    bool detect_abuse(const cv::Mat& frame, float conf_threshold=0.5) {
        // 预处理
        cv::Mat input_blob = preprocess(frame);

        // 创建输入Tensor
        std::array<int64_t, 4> input_shape{1, input_blob.channels(), input_size.height, input_size.width};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_blob.ptr<float>(),
            input_blob.total() * input_blob.channels(),
            input_shape.data(),
            input_shape.size()
        );

        // 运行推理
        auto outputs = session.Run(Ort::RunOptions{nullptr},
                                 input_names.data(), &input_tensor, 1,
                                 output_names.data(), 1);

        // 解析输出（YOLOv8输出格式为[1,84,8400]）
        float* output_data = outputs.front().GetTensorMutableData<float>();
        const auto output_shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();

        // 遍历检测结果
        const int num_classes = output_shape[1] - 4; // 84 = 4 box + 80 classes
        for (int i = 0; i < output_shape[2]; ++i) {
            float* ptr = output_data + i * output_shape[1];
            float conf = ptr[4]; // 假设类别0是abuse
            if (conf > conf_threshold) {
                return true;
            }
        }
        return false;
    }
};

class PPTSM_ONNX_Inference {
private:
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<int64_t> input_shape;

public:
    PPTSM_ONNX_Inference(const std::string& onnx_path) : env(ORT_LOGGING_LEVEL_WARNING, "PPTSM_ONNX") {
        Ort::SessionOptions session_options;
        session = Ort::Session(env, onnx_path.c_str(), session_options);

        // Get input and output names
        input_names.push_back(session.GetInputName(0, allocator));
        output_names.push_back(session.GetOutputName(0, allocator));

        // Get input shape
        auto input_info = session.GetInputTypeInfo(0);
        auto tensor_info = input_info.GetTensorTypeAndShapeInfo();
        input_shape = tensor_info.GetShape();
    }

    std::vector<float> preprocess(const std::deque<cv::Mat>& frame_buffer) {
        std::vector<cv::Mat> processed_frames;

        for (const auto& frame : frame_buffer) {
            cv::Mat resized, rgb;
            cv::resize(frame, resized, cv::Size(320, 320));
            cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
            processed_frames.push_back(rgb);
        }

        // Create a single tensor from all frames
        cv::Mat concat_frame;
        cv::vconcat(processed_frames, concat_frame);

        // Convert to float and normalize
        cv::Mat float_frame;
        concat_frame.convertTo(float_frame, CV_32F);
        float_frame /= 255.0;

        // Convert to vector<float> in CHW format
        std::vector<float> input_tensor_values;
        input_tensor_values.assign((float*)float_frame.datastart, (float*)float_frame.dataend);

        return input_tensor_values;
    }

    std::map<std::string, float> predict(const std::deque<cv::Mat>& frame_buffer) {
        auto input_tensor_values = preprocess(frame_buffer);

        // Create input tensor
        std::vector<int64_t> current_input_shape = {1, static_cast<int64_t>(frame_buffer.size()), 3, 320, 320};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            current_input_shape.data(),
            current_input_shape.size()
        );

        // Run inference
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            &input_tensor,
            1,
            output_names.data(),
            1
        );

        // Get output
        float* floatarr = output_tensors.front().GetTensorMutableData<float>();
        float fight_score = floatarr[1];  // Assuming index 1 is the fight score

        return {{"fight", std::abs(fight_score)}};
    }
};

void test_video(const std::string& input_path, const std::string& yolo_path, const std::string& pptsm_path) {
    // Parameters
    const size_t deque_maxlen = 8;
    const double frame_interval = 0.1;  // seconds
    const double show_time = 2.0;      // seconds
    const float det_conf = 0.5f;       // YOLO confidence threshold
    const float det_iou = 0.6f;        // YOLO IOU threshold
    const float cls_conf = 0.5f;       // PP-TSM confidence threshold

    // Initialize models
    PPTSM_ONNX_Inference pptsm_model(pptsm_path);
    YOLOv8_ONNX_Inference yolo_model(yolo_path);

    // Open video
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return;
    }

    // Get video properties
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    // Create video writer
    cv::VideoWriter writer("output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                          fps, cv::Size(width, height));

    // Initialize frame buffer
    std::deque<cv::Mat> frame_buffer;

    // Timers
    auto last_add_time = std::chrono::steady_clock::now();
    auto last_fight_time = std::chrono::steady_clock::now() - std::chrono::seconds(10);

    // Main loop
    cv::Mat frame;
    while (cap.read(frame)) {
        auto current_time = std::chrono::steady_clock::now();

        // Add frame to buffer at specified interval
        std::chrono::duration<double> elapsed_add = current_time - last_add_time;
        if (elapsed_add.count() >= frame_interval) {
            frame_buffer.push_back(frame.clone());
            if (frame_buffer.size() > deque_maxlen) {
                frame_buffer.pop_front();
            }
            last_add_time = current_time;
        }

        // Simulate YOLOv8 detection (replace with actual YOLOv8 C++ code)

        bool yolo_detected_fight = yolo_model.detect_abuse(frame, det_conf);

        if (yolo_detected_fight) {
            if (frame_buffer.size() < deque_maxlen) {
                last_fight_time = current_time;
                std::cout << "Detect without cls!" << std::endl;
            }

            if (frame_buffer.size() == deque_maxlen) {
                auto result = pptsm_model.predict(frame_buffer);
                std::cout << "result is: " << result["fight"] << std::endl;

                if (result["fight"] > cls_conf) {
                    std::cout << "Detect with cls!" << std::endl;
                    last_fight_time = current_time;
                }
            }
        }

        // Check if we should show "FIGHT!" text
        std::chrono::duration<double> elapsed_fight = current_time - last_fight_time;
        if (elapsed_fight.count() < show_time) {
            cv::putText(frame, "FIGHT!", cv::Point(50, 150),
                       cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255), 5);
        }

        // Write frame to output
        writer.write(frame);
    }

    // Clean up
    cap.release();
    writer.release();
}

int main() {
    // Model paths —— 均需要onnx文件
    std::string yolo_pt_path = "/civi/data/full_dataset/runs/detect/train_full-and-ours_more/weights/best.pt";
    std::string pptsm_onnx_path = "./inference/417ppTSM_finetune/ppTSM.onnx";

    // Check if model files exist (simplified check)
    // In real code, you'd want more robust file existence checking
    std::cout << "Note: This implementation doesn't actually check file existence" << std::endl;

    // Video path
    std::string input_video = "dataset/Our-more-clips_two-type/fight/fight_4_1_008.mp4";

    // Start detection
    test_video(input_video, yolo_pt_path, pptsm_onnx_path);

    return 0;
}