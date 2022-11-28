//
// Created by Joe Bowser on 2022-11-26.
//

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
