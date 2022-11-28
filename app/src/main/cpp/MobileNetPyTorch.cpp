//
// Created by Joe Bowser on 2022-11-26.
//
#include "Model.h"
#include "MobileNetPyTorch.h"
#include "MobileCallGuard.h"

bool MLStats::MobileNetPyTorch::loadModel() {
    auto qengines = at::globalContext().supportedQEngines();
    if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) !=
        qengines.end())
    {
        at::globalContext().setQEngine(at::QEngine::QNNPACK);
    }

    MobileCallGuard guard;
    if(getDevice() == MLStats::Device::GPU)
    {
        mModule = torch::jit::load(PYTORCH_PATH + "mobilenet_v2_vulkan_nhwc.pt");
    }
    else if(getDevice() == MLStats::Device::NNAPI)
    {
        mModule = torch::jit::load(PYTORCH_PATH + "mobilenetv2-quant_core-nnapi.pt");
    }
    else
    {
        // Quantized Models can live here
        if(getDataType() == MLStats::DataType::Int8)
        {
            mModule = torch::jit::load(PYTORCH_PATH + "mobilenetv2-quant_core-cpu.pt");
        }
        else
        {
            mModule = torch::jit::load(PYTORCH_PATH + "mobilenet_v2_nhwc.pt");
        }
    }
    mModule.eval();
    return true;
}


std::vector<MLStats::ResultSet> MLStats::MobileNetPyTorch::doTestRun(std::string & externalPath) {

    std::vector <MLStats::ResultSet> output;

    for(int i = 0; i < filePaths.size(); ++i) {
        MLStats::ResultSet record;
        record.framework = FRAMEWORK;
        record.device = getDeviceString();

        const auto sizes = std::vector < int64_t > {1, 3, 224, 224};
        float * blob = (float *)preProcessImage(filePaths[i]).data;


        auto stride_arr = c10::get_channels_last_strides_2d(sizes);
        auto input = torch::from_blob(
                blob,
                torch::IntArrayRef(sizes),
                torch::IntArrayRef(stride_arr),
                at::TensorOptions(at::kFloat)
                        .memory_format(at::MemoryFormat::ChannelsLast));

        std::vector <torch::jit::IValue> pytorchInputs;
        if(getDevice() == MLStats::Device::GPU && at::is_vulkan_available()) {
            auto gpuInput = input.vulkan();
            pytorchInputs.emplace_back(gpuInput);
        }
        else
        {
            pytorchInputs.emplace_back(input);
        }

        auto start = std::chrono::steady_clock::now();
        auto outputSet = [&]() {
            MobileCallGuard guard;

            return mModule.forward(pytorchInputs);
        }();
        auto end = std::chrono::steady_clock::now();

        std::chrono::duration<double> elapsed_seconds = end-start;
        record.duration = elapsed_seconds.count();

        output.push_back(record);
    }

    return output;

}
