/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

import neuro.NeuralNet
import neuro.Node.Activation
import neuro.Util
import kotlin.math.abs
import kotlin.random.Random

/**
 *
 * @author rickard
 */
object TestAddition {

    private val testData = arrayOf(

            arrayOf(floatArrayOf(1.7f, 7.6f), floatArrayOf(9.5f)), arrayOf(floatArrayOf(0.1f, 8f), floatArrayOf(8.1f)), arrayOf(floatArrayOf(1.5f, 1.7f), floatArrayOf(3.2f)), arrayOf(floatArrayOf(5f, 4f), floatArrayOf(9f)), arrayOf(floatArrayOf(0.5f, 0.1f), floatArrayOf(0.6f)), arrayOf(floatArrayOf(1f, 1f), floatArrayOf(2f)), arrayOf(floatArrayOf(2.1f, 3.1f), floatArrayOf(5.2f)), arrayOf(floatArrayOf(3f, 1.3f), floatArrayOf(4.3f)), arrayOf(floatArrayOf(2.3f, 5.3f), floatArrayOf(7.6f)), arrayOf(floatArrayOf(1.6f, 4.8f), floatArrayOf(6.4f)))

    fun test() {
        val testNet = Util.setupDecisionNet(2, 10, Activation.TANH, 1, Activation.IDENTITY)
        testNet.learningRate = 0.0003f

        val epochs = 5000
        val batchSize = 1

        val trainingData = generateTrainingData(1000, Random(42))

        for (k in 0 until epochs) {
            println("Epoch $k")
            trainingData.shuffle()
            val accumulatedOutputError = FloatArray(testNet.output!!.size)

            val accumulatedHiddenError = Array(testNet.hiddenLayers!!.size) { FloatArray(testNet.hiddenLayers!![0].size) }

            var trainingError = 0f

            for (i in 0 until trainingData.size) {

                testNet.setInput(trainingData[i][0])
                testNet.feedForward()
                val error = testNet.backprop(trainingData[i][1], i % batchSize == batchSize - 1, accumulatedOutputError, accumulatedHiddenError, batchSize)
                trainingError = trainingData[i][1][0] - testNet.output!![0].getOutput()
                if (i % batchSize == batchSize - 1) {
                    for (j in accumulatedOutputError.indices) {
                        accumulatedOutputError[j] = 0f
                    }

                    for (j in 0 until accumulatedHiddenError[0].size) {
                        accumulatedHiddenError[0][j] = 0f
                    }
                }

            }
            val testErrorRate = verify(testNet)
        }

        // verification
        verify(testNet)

    }

    private fun generateTrainingData(nSamples: Int, rand: Random): MutableList<Array<FloatArray>> {
        val trainingData = mutableListOf<Array<FloatArray>>()

        for (i in 0 until nSamples) {
            val data = Array(2) { FloatArray(2) }
            data[0][0] = rand.nextFloat() * 10f
            data[0][1] = rand.nextFloat() * (10f - data[0][0])
            data[1][0] = data[0][0] + data[0][1]
            trainingData.add(data)
        }

        return trainingData

    }

    private fun verify(net: NeuralNet): Float {
        var correctAnswers = 0
        var errorSum = 0f
        for (i in testData.indices) {
            net.setInput(testData[i][0])
            net.feedForward()

            val success = false
            if (abs(testData[i][1][0] - net.output!![0].getOutput()) < 0.1f) {
                correctAnswers++


                continue
            }
            println("Expected: " + testData[i][1][0] + " Actual: " + net.output!![0].getOutput())
            errorSum += abs(testData[i][1][0] - net.output!![0].getOutput())

        }
        println("Verify results: $correctAnswers $errorSum")
        return errorSum
    }
}
