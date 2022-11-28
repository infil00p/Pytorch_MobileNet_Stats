/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 ~ Copyright 2021 Adobe
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

package org.infil00p.pytorch_mobilenet_stats

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStream

class AssetHandler internal constructor(var mCtx: Context) {
    var LOGTAG = "AssetHandler"

    inner class ModelFileInit internal constructor(
        var mModelName: String,
        var mDataDir: File,
        var mAssetManager: AssetManager,
        var mModelFiles: Array<String>
    ) {
        var mTopLevelFolder: File? = null
        var mResourcesFolder: File? = null
        var mModelFolder: File? = null

        @Throws(IOException::class)
        private fun InitModelFiles() {
            copyModelFiles()
        }

        @Throws(IOException::class)
        private fun copyFileUtil(
            files: Array<String>,
            dir: File?
        ) {
            // For this example, we're using the internal storage
            for (file in files) {
                val inputFile = mAssetManager.open("$mModelName/$file")
                var outFile: File
                val dir = File(mDataDir.toString() + "/" +  mModelName)
                outFile = File(dir, file)
                val out: OutputStream = FileOutputStream(outFile)
                val buffer = ByteArray(1024)
                var length: Int
                while (inputFile.read(buffer).also { length = it } != -1) {
                    out.write(buffer, 0, length)
                }
                inputFile.close()
                out.flush()
                out.close()
            }
        }

        @Throws(IOException::class)
        private fun copyModelFiles() {
            copyFileUtil(mModelFiles, mModelFolder)
        }


        private fun createTopLevelDir() {
            mTopLevelFolder = File(mDataDir.absolutePath, mModelName)
            mTopLevelFolder!!.mkdir()
        }


        init {
            createTopLevelDir()
            InitModelFiles()
        }
    }

    @Throws(IOException::class)
    private fun Init() {
        val dataDirectory = mCtx.filesDir
        val assetManager = mCtx.assets

        val pytorch = ModelFileInit(
            "pytorch",
            dataDirectory,
            assetManager,
            arrayOf(
                "labels.txt",
                "mobilenet_v2.pt",
                "mobilenet_v2_nhwc.pt",
                "mobilenet_v2_vulkan_nhwc.pt",
                "mobilenet_v2_vulkan_nchw.pt",
                "mobilenetv2-quant_core-cpu.pt",
                "mobilenetv2-quant_core-nnapi.pt",
                "mobilenetv2-quant_full-cpu.pt",
                "mobilenetv2-quant_full-nnapi.pt",
            )
        )

        val ort = ModelFileInit(
            "ort",
            dataDirectory,
            assetManager,
            arrayOf(
                "mobilenet_v2.ort",
                "mobilenet_v2.with_runtime_opt.ort",
                "mobilenet_v2_static_nnapi.ort",
                "mobilenet_v2_static_nnapi.with_runtime_opt.ort"
            )
        )

        val tflite = ModelFileInit(
            "tflite",
            dataDirectory,
            assetManager,
            arrayOf(
                "model_float32.tflite",
                "model_float16.tflite"
            )
        )

        val tflite_google = ModelFileInit(
            "tflite_google",
            dataDirectory,
            assetManager,
            arrayOf(
                "lite-model_mobilenet_v2_100_224_fp32_1.tflite",
                "lite-model_mobilenet_v2_100_224_uint8_1.tflite"
            )
        )

        // This is pretty terrible, to be honest, and we could do better
        val image_set = ModelFileInit(
            "image_set",
            dataDirectory,
            assetManager,
            arrayOf(
                "01.png","02.png","03.png","04.png","05.png","06.png","07.png","08.png","09.png",
                "10.png", "11.png","12.png","13.png","14.png","15.png","16.png","17.png","18.png",
                "19.png", "20.png","21.png","22.png","23.png","24.png","25.png"
            )
        )

    }

    init {
        try {
            Init()
        } catch (e: IOException) {
            Log.d(LOGTAG, "Unable to get models from storage")
        }
    }
}