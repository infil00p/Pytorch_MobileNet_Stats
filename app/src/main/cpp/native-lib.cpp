#include <jni.h>
#include <string>

#include "MobileNet.h"
#include "MobileNetORT.h"
#include "MobileNetTFLite.h"
#include "MobileNetPyTorch.h"

extern "C"
JNIEXPORT jboolean JNICALL
Java_org_infil00p_pytorch_1mobilenet_1stats_MainActivity_doTest(JNIEnv *env, jobject thiz,
                                                                jstring external_file_path,
                                                                jint device, jint quant,
                                                                jint useGoogle) {
    auto cDevice = static_cast<MLStats::Device>(device);
    auto cType = static_cast<MLStats::DataType>(quant);
    int NUM_TFLITE_THREADS = 4;



    std::string externalPath = std::string(env->GetStringUTFChars(external_file_path, nullptr));

    // PyTorch
    {
        std::unique_ptr<MLStats::MobileNetPyTorch> pytorch = std::make_unique<MLStats::MobileNetPyTorch>(cDevice, cType);
        pytorch->loadModel();
        auto results = pytorch->doTestRun(externalPath);
        MLStats::Model::createReport(results, externalPath);
    }

    {
        std::unique_ptr<MLStats::MobileNetTFLite> tflite = std::make_unique<MLStats::MobileNetTFLite>(cDevice, cType);
        tflite->setCPUThreads(NUM_TFLITE_THREADS);
        tflite->useGoogleModels((useGoogle != 0 ));
        tflite->loadModel();
        auto results = tflite->doTestRun(externalPath);
        MLStats::Model::createReport(results, externalPath);
    }

    /*
     * ORT IS JUST TOTALLY BROKEN RIGHT NOW
    if(cDevice == MLStats::Device::CPU)
    {
        std::unique_ptr<MLStats::MobileNetORT> ort = std::make_unique<MLStats::MobileNetORT>(cDevice, cType);
        ort->loadModel();
        auto results = ort->doTestRun(externalPath);
        MLStats::Model::createReport(results, externalPath);
    }
    */


    return true;
}