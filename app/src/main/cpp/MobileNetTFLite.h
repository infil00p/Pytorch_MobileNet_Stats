//
// Created by Joe Bowser on 2022-11-26.
//

#ifndef PYTORCH_MOBILENET_STATS_MOBILENETTFLITE_H
#define PYTORCH_MOBILENET_STATS_MOBILENETTFLITE_H

#include "MobileNet.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_c_api.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "opencv2/opencv.hpp"

namespace MLStats {
    class MobileNetTFLite : public MobileNet {
    public:
        MobileNetTFLite(Device cDevice, DataType cType) : MobileNet(cDevice, cType)
        {

        }
        bool loadModel();
        static cv::Mat quantPreProcessing(const std::string& path);
        std::vector<ResultSet> doTestRun(std::string & externalPath);
        void setCPUThreads(int threads) {
            numThreads = threads;
        }
        void useGoogleModels(bool useModels) {
            googleModels = useModels;
        }
        ~MobileNetTFLite() {
            TfLiteInterpreterDelete(interpreter);
            TfLiteInterpreterOptionsDelete(mOptions);
            if (delegate != nullptr) {
                if (getDevice() == MLStats::Device::NNAPI) {
                    TfLiteNnapiDelegateDelete(delegate);
                } else {
                    TfLiteGpuDelegateV2Delete(delegate);
                }
            }
            TfLiteModelDelete(model);
        }
    protected:
        bool isNHWC=true;
    private:
        bool googleModels = false;
        //We should use four threads on TFLite for a real world scenario
        int numThreads = 4;
        TfLiteModel * model;
        TfLiteInterpreterOptions * mOptions;
        TfLiteInterpreter* interpreter;
        TfLiteDelegate* delegate = nullptr;
        std::string FRAMEWORK = "tflite";
        std::string FILE_PATH = "/data/data/org.infil00p.pytorch_mobilenet_stats/files/";
        std::string TF_PATH = FILE_PATH + "tflite/";
        std::string const TF_GOOGLE = FILE_PATH + "tflite_google/";
    };
}




#endif //PYTORCH_MOBILENET_STATS_MOBILENETTFLITE_H
