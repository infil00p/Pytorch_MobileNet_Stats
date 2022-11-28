# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.18.1)

# Declares and names the project.

project("superresMLStats")

#Grab the build and external directories

set(build_DIR ${CMAKE_SOURCE_DIR}/../../../build)
set(OpenCV_Android ${CMAKE_SOURCE_DIR}/../../../../../external/OpenCV-android-sdk)
set(rapidjson ${CMAKE_SOURCE_DIR}/../../../../../external/rapidjson)

# PyTorch Headers and Libs
file(GLOB PYTORCH_INCLUDE_DIRS "${build_DIR}/pytorch_android*.aar/headers")
file(GLOB PYTORCH_LINK_DIRS "${build_DIR}/pytorch_android*.aar/jni/${ANDROID_ABI}")

find_library(PYTORCH_LIBRARY pytorch_jni
        PATHS ${PYTORCH_LINK_DIRS}
        NO_CMAKE_FIND_ROOT_PATH)

find_library(FBJNI_LIBRARY fbjni
        PATHS ${PYTORCH_LINK_DIRS}
        NO_CMAKE_FIND_ROOT_PATH)

# Tensorflow Headers and Libs (Version MUST be specified, THANKS GOOGLE)
file(GLOB TFLITE_INCLUDE_DIRS "${build_DIR}/tensorflow-lite-2.10.0.aar/headers")
file(GLOB TFLITE_LINK_DIRS "${build_DIR}/tensorflow-lite-2.10.0.aar/jni/${ANDROID_ABI}" )
file(GLOB TFLITE_GPU_INCLUDE_DIRS "${build_DIR}/tensorflow-lite-gpu-2.10.0.aar/headers" )
file(GLOB TFLITE_GPU_LINK_DIRS "${build_DIR}/tensorflow-lite-gpu-2.10.0.aar/jni/${ANDROID_ABI}" )


# ONNX Runtime Headers and Libs
file(GLOB ORT_INCLUDE_DIRS "${build_DIR}/onnxruntime-mobile-1.13.1.aar/headers")
file(GLOB ORT_LINK_DIRS "${build_DIR}/onnxruntime-mobile-1.13.1.aar/jni/${ANDROID_ABI}" )

add_library(tflite SHARED IMPORTED)
set_target_properties(
        tflite
        PROPERTIES
        IMPORTED_LOCATION "${TFLITE_LINK_DIRS}/libtensorflowlite_jni.so")

add_library(tflite_gpu SHARED IMPORTED)
set_target_properties(
        tflite_gpu
        PROPERTIES
        IMPORTED_LOCATION "${TFLITE_GPU_LINK_DIRS}/libtensorflowlite_gpu_jni.so")

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(
        onnxruntime
        PROPERTIES
        IMPORTED_LOCATION "${ORT_LINK_DIRS}/libonnxruntime.so")

# Bring in OpenCV from an external directory
# I really wish OpenCV was distributable via AAR
add_library(lib_opencv SHARED IMPORTED)
set_target_properties(
        lib_opencv
        PROPERTIES
        IMPORTED_LOCATION ${OpenCV_Android}/sdk/native/libs/${ANDROID_ABI}/libopencv_java4.so)


include_directories(
        ${PYTORCH_INCLUDE_DIRS}
        ${TFLITE_INCLUDE_DIRS}
        ${TFLITE_GPU_INCLUDE_DIRS}
        ${ORT_INCLUDE_DIRS}
        ${OpenCV_Android}/sdk/native/jni/include
        ${rapidjson}/include
)


# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
        mobilenetstats

        # Sets the library as a shared library.
        SHARED
        MobileNetORT.cpp
        MobileNetTFLite.cpp
        MobileNetPyTorch.cpp
        MobileNet.cpp
        Model.cpp
        # Provides a relative path to your source file(s).
        native-lib.cpp)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        mobilenetstats
        ${FBJNI_LIBRARY}
        ${PYTORCH_LIBRARY}
        tflite_gpu
        tflite
        onnxruntime
        lib_opencv
        # Links the target library to the log library
        # included in the NDK.
        ${log-lib})