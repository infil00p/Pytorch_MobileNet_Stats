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
