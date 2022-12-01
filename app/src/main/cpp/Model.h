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

#ifndef SUPERRESSTATS_MODEL_H
#define SUPERRESSTATS_MODEL_H

#include <string>
#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"

namespace MLStats
{
    // We support all the quantizations
    enum DataType {
        Float32 = 0,
        Float16 = 1,
        Int8 = 2
    };

    // This is on Android and we intend to support all the accelerators
    enum Device {
        CPU = 0,
        GPU = 1,
        NNAPI = 2,
        SNPE = 3,
    };

    // This needs to contain the
    struct ResultSet {
        std::string imageUri;
        double duration;
        std::string framework;
        std::string device;
        std::string dataType;
    };

    /*
     * Abstraction for running the model in the harness
     * The expectation is that this class will be overriden for pre and post processing code
     * and the modelPath is optional
     */

    class Model {
    public:
        Model(Device device, DataType type) {
            mDataType = type;
            mDevice = device;
        }
        static void createReport(std::vector<MLStats::ResultSet> & results, std::string & externalPath);
        Device getDevice() { return mDevice; }
        DataType getDataType() { return mDataType; }
        std::string getDeviceString();
    protected:
        DataType mDataType;
        Device mDevice;
    };

}


#endif //SUPERRESSTATS_MODEL_H
