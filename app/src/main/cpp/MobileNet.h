//
// Created by Joe Bowser on 2022-11-26.
//

#ifndef PYTORCH_MOBILENET_STATS_MOBILENET_H
#define PYTORCH_MOBILENET_STATS_MOBILENET_H

#include <string>
#include <opencv2/opencv.hpp>
#include "Model.h"

namespace MLStats {


    class MobileNet : public Model {
    public:
        MobileNet(Device cDevice, DataType cType);

        virtual bool loadModel() = 0;
        cv::Mat preProcessImage(std::string path);

        std::vector <std::string> filePaths;

        std::string IMAGE_PATH = "/data/data/org.infil00p.pytorch_mobilenet_stats/files/image_set/";
    protected:
        bool isNHWC = false;
    };

}

#endif //PYTORCH_MOBILENET_STATS_MOBILENET_H
