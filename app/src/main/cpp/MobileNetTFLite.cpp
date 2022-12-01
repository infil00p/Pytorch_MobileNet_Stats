/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 ~ Copyright 2022 Adobe
 ~
 ~ Licensed under the Apache License, Version 2.0 (the "License");
 ~ you may not use this file except in compliance with the License.
 ~ You may obtain a copy of the License at
 ~
 ~     http://www.apache.org/licenses/LICENSE-2.0
 ~
 ~ Unless required by applicable law or agreed to in writing, software
 ~ distributed under the License is distributed on an "AS IS" BASIS,
 ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 ~ See the License for the specific language governing permissions and
 ~ limitations under the License.
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#include "MobileNetTFLite.h"

#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/delegates/gpu/delegate.h>
#include <tensorflow/lite/delegates/nnapi/nnapi_delegate_c_api.h>
#include <chrono>
#include <android/log.h>

#define PRE_TAG "SuperRes_TFLite"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,    PRE_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,     PRE_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,     PRE_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,    PRE_TAG, __VA_ARGS__)

bool MLStats::MobileNetTFLite::loadModel() {
    std::string fullPath;
    mOptions = TfLiteInterpreterOptionsCreate();

    if(getDevice() == MLStats::Device::NNAPI)
    {
        TfLiteNnapiDelegateOptions options = TfLiteNnapiDelegateOptionsDefault();
        delegate = TfLiteNnapiDelegateCreate(&options);
        TfLiteInterpreterOptionsAddDelegate(mOptions, delegate);

        //NNAPI is only going to use int8 models from Google, no playing around
        fullPath = TF_GOOGLE + "lite-model_mobilenet_v2_100_224_uint8_1.tflite";
        model = TfLiteModelCreateFromFile(fullPath.c_str());
    }
    else {
        if (getDevice() == MLStats::Device::GPU) {
            TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
            delegate = TfLiteGpuDelegateV2Create(&options);
            TfLiteInterpreterOptionsAddDelegate(mOptions, delegate);
        }
        else
        {
            // Set the number of threads on the CPU
            TfLiteInterpreterOptionsSetNumThreads(mOptions,numThreads);
        }
        if(googleModels)
        {
            // I don't know if we should use FP16 for GPU, I feel like that would give the advantage
            // to Google, and I don't know what the pre/post on FP16 even looks like here
            fullPath = TF_GOOGLE + "lite-model_mobilenet_v2_100_224_fp32_1.tflite";
        }
        else
        {
            // This is our conversion in the use case where we have to go from PyTorch to TF
            // It's interesting that the Google models came out of IREE and were compiled into TF
            // flatbuffers.  Can we do this with PyTorch?
            fullPath = TF_PATH + "model_float32.tflite";
        }
        model = TfLiteModelCreateFromFile(fullPath.c_str());
    }

    return model != nullptr;
}

std::vector <MLStats::ResultSet> MLStats::MobileNetTFLite::doTestRun(std::string &externalPath) {
    std::vector <ResultSet> output;

    // Create the interpreter
    interpreter = TfLiteInterpreterCreate(model, mOptions);
    // Allocate the tensors to warm up the model
    TfLiteInterpreterAllocateTensors(interpreter);
    for(size_t i = 0; i < filePaths.size(); ++i) {
        // Do the Open CV thing
        ResultSet record;
        record.framework = "TFLite";
        record.device = getDeviceString();
        cv::Mat inputMat;
        size_t imageSize = inputMat.rows * inputMat.cols * inputMat.channels();

        if(mDataType == MLStats::Int8)
        {
            inputMat = quantPreProcessing(filePaths[i]);
        }
        else
        {
            inputMat = preProcessImage(filePaths[i]);
            imageSize = imageSize * sizeof(float);
        }

        TfLiteTensor * input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
        std::string name_of_tensor = std::string(TfLiteTensorName(input_tensor));
        // Copy the data over (THIS IS WHERE TF IS BAD)
        if(inputMat.data == nullptr) {
            LOGE("We don't have data");
        }
        TfLiteTensorCopyFromBuffer(input_tensor, inputMat.data, imageSize);

        auto start = std::chrono::steady_clock::now();
        TfLiteInterpreterInvoke(interpreter);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        record.duration = elapsed_seconds.count();

        output.push_back(record);
    }
    return output;
}

cv::Mat MLStats::MobileNetTFLite::quantPreProcessing(const std::string& path) {
    int width  = 224;
    int height = 224;
    cv::Mat imageBGR = cv::imread(path);
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage;
    cv::resize(imageBGR,
               resizedImageBGR,
               cv::Size(width, height),
               cv::InterpolationFlags::INTER_CUBIC);
    cv::cvtColor(resizedImageBGR,
                 resizedImageRGB,
                 cv::ColorConversionCodes::COLOR_BGR2RGB);
    // Do not convert this to float32
    return resizedImageRGB;
}
