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

#include "MobileNetORT.h"

bool MLStats::MobileNetORT::loadModel() {
    std::string full_model_path = ORT_PATH + "mobilenet_v2.with_runtime_opt.ort";
    session = std::make_unique<Ort::Session>(env, full_model_path.c_str(), session_options);
    return true;
}

std::vector <MLStats::ResultSet> MLStats::MobileNetORT::doTestRun(std::string &externalPath) {
    std::vector <ResultSet> output;
    // Allocate the tensors to warm up the model
    for(int i = 0; i < filePaths.size(); ++i)
    {
        // Do the Open CV thing
        std::vector<cv::Mat> yCrCb;
        cv::Mat inputMat = preProcessImage(filePaths[i]);
        ResultSet record;
        record.framework = "ORT";

        // Allocate the things
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        constexpr size_t input_tensor_size = 224 * 224 * 3;
        std::vector<int64_t> input_node_dims = {1, 3, 224, 224};
        auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float *)inputMat.data,
                                                            input_tensor_size,
                                                            input_node_dims.data(), 4);
        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};

        auto start = std::chrono::steady_clock::now();
        auto output_result = session->Run(Ort::RunOptions{nullptr}, input_names,
                                          &input_tensor, 1,
                                          output_names, 1);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        record.duration = elapsed_seconds.count();

        record.device = getDeviceString();
        output.push_back(record);
    }
    return output;
}
