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
 
#include "Model.h"
#include "rapidjson/ostreamwrapper.h"
#include <fstream>
#include <vector>


void MLStats::Model::createReport(std::vector<MLStats::ResultSet> & results,
                                  std::string & externalPath) {
    rapidjson::Document d;
    d.SetObject();
    rapidjson::Value a(rapidjson::kArrayType);

    // Yeah, this is the same, but I get compile errors when I use this elsewhere

    // This will be identical throughout
    std::string frameworkResult = results[0].framework;

    for(auto result : results)
    {
        rapidjson::Value o(rapidjson::kObjectType);
        rapidjson::Value framework(rapidjson::kStringType);
        rapidjson::Value path(rapidjson::kStringType);
        rapidjson::Value device(rapidjson::kStringType);
        rapidjson::Value duration(rapidjson::kNumberType);
        framework.SetString(result.framework.c_str(), result.framework.size(), d.GetAllocator());
        path.SetString(result.imageUri.c_str(), result.imageUri.size(), d.GetAllocator());
        device.SetString(result.device.c_str(), result.device.size(), d.GetAllocator());
        duration.SetDouble(result.duration);
        o.AddMember("framework", framework, d.GetAllocator());
        o.AddMember("imagePath", path, d.GetAllocator());
        o.AddMember("duration", duration, d.GetAllocator());
        o.AddMember("device", device, d.GetAllocator());
        a.PushBack(o, d.GetAllocator());
    }
    d.AddMember("results", a, d.GetAllocator());

    std::string jsonPath = externalPath + "/results_" + frameworkResult + ".json";
    std::ofstream out { jsonPath };
    if(!out.is_open())
    {
        // Return out of here
        return;
    }

    // Write the JSON to a file
    rapidjson::OStreamWrapper osw {out};
    rapidjson::PrettyWriter<rapidjson::OStreamWrapper> fileWriter {osw};
    d.Accept(fileWriter);
}

std::string MLStats::Model::getDeviceString() {
    switch(mDevice) {
        case CPU: return "CPU";
        case GPU: return "GPU";
        case NNAPI: return "NNAPI";
        case SNPE: return "SNPE";
    }
}
