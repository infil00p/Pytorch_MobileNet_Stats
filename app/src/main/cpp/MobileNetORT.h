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

#ifndef PYTORCH_MOBILENET_STATS_MOBILENETORT_H
#define PYTORCH_MOBILENET_STATS_MOBILENETORT_H
#include "onnxruntime_cxx_api.h"
#include "MobileNet.h"

namespace MLStats {

    class MobileNetORT : public MobileNet {
    public:
        MobileNetORT(Device cDevice, DataType cType) : MobileNet(cDevice, cType)
        {

        }
        ~MobileNetORT() {
            if(session != nullptr)
            {
                // Reset the session before closing
                session.reset();
            }
        }
        bool loadModel();
        std::vector<ResultSet> doTestRun(std::string & externalPath);
    private:
        Ort::Env env;
        std::unique_ptr<Ort::Session> session;
        Ort::SessionOptions session_options;
        std::string FRAMEWORK = "ort";
        std::string ORT_PATH = "/data/data/org.infil00p.pytorch_mobilenet_stats/files/ort/";
    };

}




#endif //PYTORCH_MOBILENET_STATS_MOBILENETORT_H
