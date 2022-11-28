package org.infil00p.superresstats

import android.util.JsonReader
import java.io.StringReader
import java.util.*

class ResultParser {

    fun loadResultsFromJson(results : String) : ResultSet
    {
        var reader = JsonReader(StringReader(results))
        var resultVector : Vector<Result>
        try {
            reader.beginObject()
            resultVector = readResults(reader)
        } finally {
            reader.close()
        }
        return ResultSet(resultVector)
    }

    private fun readResults(reader: JsonReader) : Vector<Result>
    {
        // Create empty vector
        var resultVector = Vector<Result>()
        // Consume the results object.
        if(reader.nextName().equals("results")) {
            reader.beginArray()
            while (reader.hasNext()) {
                lateinit var framework: String
                lateinit var filename: String
                lateinit var device: String
                var duration: Double
                duration = 0.0
                reader.beginObject()
                while (reader.hasNext()) {
                    var name = reader.nextName()
                    if (name.equals("framework")) {
                        framework = reader.nextString()
                    } else if (name.equals("duration")) {
                        duration = reader.nextDouble()
                    } else if (name.equals("imagePath")) {
                        filename = reader.nextString()
                    } else if (name.equals("device")) {
                        device = reader.nextString()
                    } else {
                        reader.skipValue()
                    }
                }
                reader.endObject()
                val result = Result(framework, duration, filename, device)
                resultVector.add(result)
            }
        }
        return resultVector
    }
}