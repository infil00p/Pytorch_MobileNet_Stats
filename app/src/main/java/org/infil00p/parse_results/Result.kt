package org.infil00p.superresstats

import java.util.*

class Result(val framework: String, val duration: Double, val imageUri: String, val device: String ) {

}

class ResultSet(val resultList: Vector<Result>)
{
    fun calculateAverage() : Double
    {
        var calcList = resultList;
        // Remove the first before calculating the average.
        calcList.removeFirst();
        var total = 0.0;
        for(result in calcList) {
            total += result.duration;
        }
        return total/calcList.size;
    }

    fun getMin() : Double
    {
        var lowestNum = resultList.firstElement().duration;
        for(result in resultList){
            if(lowestNum > result.duration)
                lowestNum = result.duration
        }
        return lowestNum
    }

    fun getMax() : Double
    {
        var highestNum = resultList.firstElement().duration;
        for(result in resultList){
            if(highestNum < result.duration)
                highestNum = result.duration
        }
        return highestNum
    }

}