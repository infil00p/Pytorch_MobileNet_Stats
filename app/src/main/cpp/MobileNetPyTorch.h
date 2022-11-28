//
// Created by Joe Bowser on 2022-11-26.
//

#ifndef PYTORCH_MOBILENET_STATS_MOBILENETPYTORCH_H
#define PYTORCH_MOBILENET_STATS_MOBILENETPYTORCH_H

#include "MobileNet.h"
#include "torch/script.h"
#include "opencv2/opencv.hpp"
#include "MobileCallGuard.h"

namespace MLStats {

    class MobileNetPyTorch : public MobileNet {
    public:
        MobileNetPyTorch(Device cDevice, DataType cType) : MobileNet(cDevice, cType) {
        }
        bool loadModel();
        std::vector <ResultSet> doTestRun(std::string &externalPath);
    private:
        mutable torch::jit::script::Module mModule;
        std::string FRAMEWORK="pytorch";
        bool isNHWC = true;
    };

}

#endif //PYTORCH_MOBILENET_STATS_MOBILENETPYTORCH_H
