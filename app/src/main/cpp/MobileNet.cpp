//
// Created by Joe Bowser on 2022-11-26.
//

#include "MobileNet.h"

#include <android/log.h>
#include <fstream>
#define PRE_TAG "MobileNet_Pre_Post"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,    PRE_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,     PRE_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,     PRE_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,    PRE_TAG, __VA_ARGS__)

namespace MLStats {

    cv::Mat MobileNet::preProcessImage(std::string path) {
        int width  = 224;
        int height = 224;

        cv::Mat imageBGR = cv::imread(path);

        // Since this is coming from PyTorch, we need to follow PyTorch's
        // procedure.
        cv::Mat resizedImageBGR, resizedImageRGB, resizedImage;
        cv::resize(imageBGR,
                   resizedImageBGR,
                   cv::Size(width, height),
                   cv::InterpolationFlags::INTER_CUBIC);
        cv::cvtColor(resizedImageBGR,
                     resizedImageRGB,
                     cv::ColorConversionCodes::COLOR_BGR2RGB);
        resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

        cv::Mat channels[3];
        cv::split(resizedImage, channels);
        // Normalization per channel
        // Normalization parameters obtained from
        // https://github.com/onnx/models/tree/master/vision/classification/squeezenet
        channels[0] = (channels[0] - 0.485) / 0.229;
        channels[1] = (channels[1] - 0.456) / 0.224;
        channels[2] = (channels[2] - 0.406) / 0.225;
        cv::merge(channels, 3, resizedImage);
        cv::Mat preprocessedMat;
        if (isNHWC)
        {
            //TODO: Replace with cv::Mat::Transpose, we don't need a whole ML framework to do this
            cv::dnn::blobFromImage(resizedImage, preprocessedMat);
        }
        else
        {
            preprocessedMat = resizedImage;
        }

        return preprocessedMat;
    }


    MobileNet::MobileNet(Device cDevice, DataType cType) : Model(cDevice, cType) {
        for (int i = 1; i < 26; ++i) {
            std::string current_file;
            if (i < 10) {
                current_file = IMAGE_PATH + "0" + std::to_string(i) + ".png";
            } else {
                current_file = IMAGE_PATH + std::to_string(i) + ".png";
            }
            filePaths.push_back(current_file);
        }
    }

}
