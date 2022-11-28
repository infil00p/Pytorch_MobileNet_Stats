//
// Created by Joe Bowser on 2022-11-25.
//

#ifndef SUPERRESSHOOTOUT_MODEL_H
#define SUPERRESSHOOTOUT_MODEL_H

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


#endif //SUPERRESSHOOTOUT_MODEL_H
