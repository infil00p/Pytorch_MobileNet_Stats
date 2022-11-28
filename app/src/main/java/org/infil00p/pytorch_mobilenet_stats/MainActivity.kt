package org.infil00p.pytorch_mobilenet_stats

import android.R
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.TextView
import android.widget.Toast
import androidx.core.content.ContextCompat
import androidx.core.view.forEach
import org.infil00p.pytorch_mobilenet_stats.databinding.ActivityMainBinding
import org.infil00p.superresstats.ResultParser
import org.infil00p.superresstats.ResultSet
import java.io.File
import java.util.*
import kotlin.collections.ArrayList
import com.github.mikephil.charting.data.BarData
import com.github.mikephil.charting.data.BarDataSet
import com.github.mikephil.charting.data.BarEntry


class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    var deviceType: Int = 0;
    var quantType: Int = 0;
    var useGoogleModels: Int = 0;

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // I'm assuming this does something
        var handler = AssetHandler(this);

        // Prep external storage permissions
        var external = this.getExternalFilesDir(null)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Disable Float16 for now
        binding.float16Radio.isEnabled = false;

        binding.button.setOnClickListener {
            if (external != null) {
                // Disable the button until we're done
                binding.button.isEnabled = false
                if (doTest(external.absolutePath, deviceType, quantType, useGoogleModels)) {
                    // Get the file from internal storage
                    val tflite_results_file = File(external, "results_tflite.json")
                    val pytorch_results_file = File(external, "results_pytorch.json")

                    val tfLiteText = tflite_results_file.readText();
                    val pytorchText = pytorch_results_file.readText();

                    // Parse these into JSON
                    val parser = ResultParser()
                    var resultList = Vector<ResultSet>()
                    resultList.addElement(parser.loadResultsFromJson(tfLiteText))
                    resultList.addElement(parser.loadResultsFromJson(pytorchText))

                    /*
                    if(deviceType == 0) {
                        val ort_results_file = File(external, "results_ort.json")
                        val ortText = ort_results_file.readText();
                        resultList.addElement(parser.loadResultsFromJson(ortText))
                    }
                    */

                    drawResults(resultList);

                    binding.button.isEnabled = true
                }
            }
        }

        // Set the default state of the API
        binding.fwGroup.setOnCheckedChangeListener { group, checkedId ->
            if(checkedId == binding.tfLiteRadio.id)
            {
                binding.accelGroup.forEach { view -> view.isEnabled = true }
                binding.quantGroup.forEach { view -> view.isEnabled = true }
                binding.float16Radio.isEnabled = false;

            }
            else if(checkedId == binding.pyTorchRadio.id)
            {
                binding.accelGroup.forEach { view -> view.isEnabled = true }
                binding.quantGroup.forEach { view -> view.isEnabled = true }
                binding.float16Radio.isEnabled = false;
            }
            else
            {
                // The ORT build that we have is actually broken.  This test
                // is for published libraries on Gradle only
                binding.accelGroup.forEach { view -> view.isEnabled = false }
                binding.quantGroup.forEach { view -> view.isEnabled = false }
                val toast = Toast.makeText(this, "ONNX Runtime AAR only supports CPU", Toast.LENGTH_SHORT)
                toast.show()
                deviceType = 0;
            }
        }

        binding.accelGroup.setOnCheckedChangeListener { group, checkedId ->
            if(checkedId == binding.nnapiRadio.id)
            {
                deviceType = 2
            }
            else if (checkedId == binding.gpuRadio.id)
            {
                deviceType = 1
            }
            else
            {
                deviceType = 0
            }
        }

        binding.quantGroup.setOnCheckedChangeListener { group, checkedId ->
            if(checkedId == binding.nnapiRadio.id)
            {
                quantType = 2
            }
            else if (checkedId == binding.gpuRadio.id)
            {
                quantType = 1
            }
            else
            {
                quantType = 0
            }
        }

        binding.tfliteGroup.setOnCheckedChangeListener { group, checkedId ->
            if(checkedId == binding.googleRadio.id) {
                useGoogleModels = 1
            }
            else
            {
                useGoogleModels = 0
            }
        }
    }

    fun drawResults(resultList : Vector<ResultSet>) {
        var barChart = binding.resultsBarChart
        val ourBarEntries: ArrayList<BarEntry> = ArrayList()

        val colors = ArrayList<Int>()
        colors.add(ContextCompat.getColor(this, R.color.holo_orange_dark))
        colors.add(ContextCompat.getColor(this, R.color.holo_red_dark))

        val tfLiteMean = resultList[0].calculateAverage().toFloat()
        val pyTorchMean = resultList[1].calculateAverage().toFloat()
        val tfliteBarEntry = BarEntry(0.0f, tfLiteMean)
        val pyTorchBarEntry = BarEntry(1.0f, pyTorchMean)
        ourBarEntries.add(tfliteBarEntry)
        ourBarEntries.add(pyTorchBarEntry)

        // ONNX Runtime is an optional value
        if(resultList.size == 3)
        {
            colors.add(ContextCompat.getColor(this, R.color.holo_blue_dark))
            val ortMean = resultList[2].calculateAverage().toFloat()
            val ortBarEntry = BarEntry(2.0f, ortMean)
            ourBarEntries.add(ortBarEntry)
        }

        val averageDataset = BarDataSet(ourBarEntries, "Average Inference Times")
        averageDataset.isHighlightEnabled = true;


        averageDataset.colors = colors

        barChart.data = BarData(averageDataset)

        val legend = barChart.legend
        legend.isEnabled = true

        barChart.invalidate()
    }
    /**
     * A native method that is implemented by the 'superresshootout' native library,
     * which is packaged with this application.
     */
    external fun doTest(externalFilePath: String, device: Int, quant: Int, useGoogle: Int): Boolean

    companion object {
        // Used to load the 'superresshootout' library on application startup.
        init {
            System.loadLibrary("mobilenetstats")
        }
    }
}