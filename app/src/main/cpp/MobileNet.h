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
