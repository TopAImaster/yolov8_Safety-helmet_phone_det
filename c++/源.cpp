#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include<vector>
#include <iostream>
#include <fstream>



static const int BatchSize = 1;
static const int INPUT_H = 2048;
static const int INPUT_W = 2048;
static const int INPUT_C = 3;


std::vector<std::string> readClassNames(const std::string& filename) {
    std::vector<std::string> classNames;

    std::ifstream file(filename);
    if (file.is_open()) {
        std::string className;
        while (std::getline(file, className)) {
            classNames.push_back(className);
        }
        file.close();
    }
    else {
        std::cout << "Failed to open file: " << filename << std::endl;
    }

    return classNames;
}

int main() {

    std::vector<std::string> labels = readClassNames("hat_phone.txt");

    ov::Core core;
    ov::CompiledModel compiled_model = core.compile_model("hat_phone_smi3.onnx","CPU");
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    // Get input port for model with one input
    //auto input_port = compiled_model.input();

    auto input = infer_request.get_input_tensor(0);
    auto output = infer_request.get_output_tensor(0);


    input.set_shape({BatchSize, INPUT_H, INPUT_W, INPUT_C });
    float* input_data_host = input.data<float>();


    std::string vediopath;
    std::cout << "请输入绝对路径：";
    std::getline(std::cin, vediopath);



    //打开视频文件
    cv::VideoCapture inputVideo(vediopath);

    // 检查视频文件是否成功打开
    if (!inputVideo.isOpened()) {
        std::cout << "无法打开输入视频文件" << std::endl;
        return -1;
    }

    // 获取输入视频的相关信息
    int frameWidth = inputVideo.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHeight = inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = inputVideo.get(cv::CAP_PROP_FPS);

    // 创建输出视频文件
    cv::VideoWriter outputVideo("output_video.mp4", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, cv::Size(frameWidth, frameHeight));
    int totalFrames = inputVideo.get(cv::CAP_PROP_FRAME_COUNT);
    // 检查输出视频文件是否成功创建
    if (!outputVideo.isOpened()) {
        std::cout << "无法创建输出视频文件" << std::endl;
        return -1;
    }
    //以上打开视频文件

    cv::namedWindow("Progress");

    // 创建进度条
    int currentFrame = 0;
    cv::createTrackbar("Progress", "Progress", &currentFrame, totalFrames);

   // auto image = cv::imread("C:\\Users\\12204\\Desktop\\asd.png");

    cv::Mat image;
    while (inputVideo.read(image)) {
        // 在这里进行帧处理操作
        // ...
        int originalWidth = image.cols;
        int originalHeight = image.rows;

        // 目标图像大小
        int targetSize = 2048;

        // 计算缩放比例
        float scale = static_cast<float>(targetSize) / std::max(originalWidth, originalHeight);

        // 计算缩放后的图像尺寸
        int targetWidth = static_cast<int>(originalWidth * scale);
        int targetHeight = static_cast<int>(originalHeight * scale);

        // 计算填充的宽度和高度
        int paddingWidth = targetSize - targetWidth;
        int paddingHeight = targetSize - targetHeight;

        // 计算左上角填充量
        int offsetX = paddingWidth / 2;
        int offsetY = paddingHeight / 2;

        // 定义变换矩阵
        cv::Mat M = cv::Mat::zeros(2, 3, CV_32FC1);

        // 设置缩放和平移参数
        M.at<float>(0, 0) = scale;
        M.at<float>(1, 1) = scale;
        M.at<float>(0, 2) = offsetX;
        M.at<float>(1, 2) = offsetY;


        cv::Mat img;

        // 进行仿射变换
        cv::Mat transformedImage;
        cv::warpAffine(image, transformedImage, M, cv::Size(2048, 2048), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));


        int image_area = transformedImage.cols * transformedImage.rows;
        unsigned char* pimage = transformedImage.data;//返回指向数据的指针
        float* phost_b = input_data_host + image_area * 0;
        float* phost_g = input_data_host + image_area * 1;
        float* phost_r = input_data_host + image_area * 2;
        for (int i = 0; i < image_area; ++i, pimage += 3) {
            // 注意这里的顺序rgb调换了
            *phost_r++ = pimage[0] / 255.0f;
            *phost_g++ = pimage[1] / 255.0f;
            *phost_b++ = pimage[2] / 255.0f;
        }

        infer_request.infer();

        const float* output_buffer = output.data<const float>();


        cv::Mat dout(86016, 7, CV_32F, (float*)output_buffer);//将一串数组编程 mat类型的（8400*84）


        std::vector<cv::Rect> boxes;//std::vector<cv::Rect> 表示一个存储了 cv::Rect 对象的动态数组，其中 cv::Rect 是 OpenCV 库中的一个类，用于表示矩形区域。
        std::vector<int> classIds;
        std::vector<float> confidences;

        for (int i = 0; i < dout.rows; i++) {
            cv::Mat classes_scores = dout.row(i).colRange(4, 7);//对 dout 的第 i 行进行操作，并提取了列索引从 4 到 84 的子区域。
            cv::Point classIdPoint;
            double score;
            minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

            // 置信度 0～1之间
            if (score > 0.25)
            {
                float cx = dout.at<float>(i, 0);
                float cy = dout.at<float>(i, 1);
                float ow = dout.at<float>(i, 2);
                float oh = dout.at<float>(i, 3);



                int x = static_cast<int>((cx - 0.5 * ow));//(ccx - 0.5 * ow)中心坐标转左上角坐标
                int y = static_cast<int>((cy - 0.5 * oh));
                int width = static_cast<int>(ow);
                int height = static_cast<int>(oh);




                cv::Rect box;
                box.x = x;
                box.y = y;
                box.width = width;
                box.height = height;

                boxes.push_back(box);
                classIds.push_back(classIdPoint.x);
                confidences.push_back(score);
            }
        }

        //NMS
        std::vector<int> indexes;
        cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
        for (size_t i = 0; i < indexes.size(); i++) {
            int index = indexes[i];
            int idx = classIds[index];
            float confi = confidences[index];

            cv::rectangle(transformedImage, boxes[index], cv::Scalar(0, 0, 255), 2, 8);//画的框

            cv::rectangle(transformedImage, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
                cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);//画的类别背景

            putText(transformedImage, labels[idx], cv::Point(boxes[index].tl().x, boxes[index].tl().y), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);//写的类别

            putText(transformedImage, std::to_string(confi).substr(0,4), cv::Point(boxes[index].tl().x, boxes[index].tl().y+20), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 255, 0), 2, 8);//写的置信度


        }


        cv::Mat inverseM;
        cv::invertAffineTransform(M, inverseM);
        cv::warpAffine(transformedImage, image, inverseM, cv::Size(originalWidth, originalHeight), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        cv::putText(image, "FPS: " + std::to_string(currentFrame), cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        // 将处理后的帧写入输出视频文件
        outputVideo.write(image);

        currentFrame++;
        cv::setTrackbarPos("Progress", "Progress", currentFrame);

        // 显示当前帧
        //cv::imshow("Frame", image);
        //cv::waitKey(0.0001); // 按需调整等待时间，用于适应视频的帧率
    }
    inputVideo.release();
    outputVideo.release();

    std::cout << "END!" << std::endl;


    // Create tensor from external memory
    //ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape());
    // Set input tensor for model with one input
    //infer_request.set_input_tensor(input_tensor);
    //infer_request.start_async();
    //infer_request.wait();
    //auto output = infer_request.get_tensor("tensor_name");
    //const float * output_buffer = output.data<const float>();



    return 0;

}

