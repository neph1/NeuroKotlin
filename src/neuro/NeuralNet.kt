/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuro

import kotlin.math.exp


/**
 *
 * @author rickard
 */
class NeuralNet {
    var input: Array<Node>? = null
    var hiddenLayers: Array<Array<Node>>? = null
    var output: Array<Node>? = null
    private var softMax = false
    private var strategy: Strategy? = null
    var learningRate = 0.18f
    private var l2Lambda = 0f
    private var learningRateDecay = 0f
    private var momentum = 0f


    fun feedForward() {
        for (n in input!!) {
            n.output = n.input
        }
        for (layer in hiddenLayers!!) {
            for (n in layer) {
                n.updateOutput()
            }
        }
        for (n in output!!) {
            n.updateOutput()
        }
    }

    fun setInput(index: Int, value: Float) {
        input!![index].setInput(value)
    }

    fun setInput(value: FloatArray) {
        for (i in input!!.indices) {
            input!![i].setInput(value[i])
        }
    }

    fun updateOutputWeights(outputError: FloatArray) {
        for (i in output!!.indices) {
            output!![i].updateWeights(learningRate, outputError[i], l2Lambda, momentum)
        }
    }

    fun getOutputLayerError(targetValues: FloatArray): FloatArray {
        val outputError = FloatArray(output!!.size)
        for (i in output!!.indices) {
            outputError[i] = output!![i].getError(targetValues[i])
            //            outputError[i] = (float) Math.pow(targetValues[i] - output[i].output, 2);
        }
        return outputError

    }

    fun backprop(trainingDataAnswer: FloatArray, updateWeights: Boolean, accumulatedOutputError: FloatArray?, accumulatedHiddenError: Array<FloatArray>?, batchSize: Int): FloatArray {
        var outputError = getOutputLayerError(trainingDataAnswer)

        if (accumulatedOutputError != null) {
            for (i in accumulatedOutputError.indices) {
                accumulatedOutputError[i] += outputError[i]
            }
            if (updateWeights) {

                updateOutputWeights(accumulatedOutputError)
            }
        }
        for (j in hiddenLayers!!.size - 1 downTo -1 + 1) {
            outputError = getHiddenLayerError(j, outputError)
            if (accumulatedHiddenError != null) {
                for (i in 0 until accumulatedHiddenError[j].size) {
                    accumulatedHiddenError[j][i] += outputError[i]
                }
            }
            if (updateWeights) {
                updateHiddenWeights(j, accumulatedHiddenError!![j])
            }
        }
        //        if(updateWeights){
        //            for (int i = 0; i < output.length; i++) {
        //                output[i].applyWeights();
        //            }
        //            for (int i = 0; i < hiddenLayers[0].length; i++) {
        //                hiddenLayers[0][i].applyWeights();
        //            }
        //        }
        return outputError
    }

    fun getError(trainingDataAnswer: FloatArray, updateWeights: Boolean): FloatArray {
        var outputError = getOutputLayerError(trainingDataAnswer)
        if (updateWeights) {
            updateOutputWeights(outputError)
        }
        for (j in hiddenLayers!!.size - 1 downTo -1 + 1) {
            outputError = getHiddenLayerError(j, outputError)
            if (updateWeights) {
                updateHiddenWeights(j, outputError)
            }
        }
        return outputError
    }

    fun getErrorDerivate(trainingDataAnswer: FloatArray, updateWeights: Boolean): FloatArray {
        val outputError = getOutputLayerError(trainingDataAnswer)
        if (updateWeights) {
            updateOutputWeights(outputError)
        }
        var j = hiddenLayers!!.size - 1
        while (j == 0) {
            val hiddenError = getHiddenLayerError(j, outputError)
            if (updateWeights) {
                updateHiddenWeights(j, hiddenError)
            }
            j--
        }

        return outputError
    }

    fun updateHiddenWeights(layer: Int, error: FloatArray) {
        val length = hiddenLayers!![layer].size
        for (i in 0 until length) {
            hiddenLayers!![layer][i].updateWeights(learningRate, error[i], l2Lambda, momentum)
        }
    }

    fun getHiddenLayerError(layer: Int, outputError: FloatArray): FloatArray {
        val length = hiddenLayers!![layer].size
        val error = FloatArray(length) // outConnection.error * output weight
        for (i in 0 until length) {
            val hidden = hiddenLayers!![layer][i]
            for (j in 0 until hidden.outConnections!!.size) {
                val n = hidden.outConnections!![j]
                error[i] += outputError[j] * n.weights!![i]
            }

            error[i] = hidden.getHiddenError(error[i])
        }
        return error
    }

    /**
     * Specialized method for doing classification. Do not use for regression
     * @param input
     * @return
     */
    fun getClassForInput(input: FloatArray): Int {
        setInput(input)
        feedForward()
        var selectedResult = -1
        if (softMax) {
            var max = 0f
            for (n in output!!) {
                if (n.output > max) {
                    max = n.output
                }
            }
            var totalInput = 0f
            for (n in output!!) {
                totalInput += exp(n.input - max)
            }
            val softMaxOutput = FloatArray(output!!.size)
            for (i in output!!.indices) {
                softMaxOutput[i] = (exp(output!![i].input - max) / totalInput) as Float
            }

            val rand = Util.random.nextFloat() as Float
            selectedResult = Util.randomDistributionSelection(softMaxOutput, rand)
        } else {
            var highestOutput = Float.NEGATIVE_INFINITY
            for (i in output!!.indices) {
                if (output!![i].output > highestOutput) {
                    highestOutput = output!![i].output
                    selectedResult = i
                }
            }
        }


        return selectedResult
    }

    fun setSoftMax(softMax: Boolean) {
        this.softMax = softMax
    }

    fun getOutput(): FloatArray {
        val out = FloatArray(output!!.size)
        for (i in output!!.indices) {
            out[i] = output!![i].output
        }
        return out
    }

    fun update() {

        if (strategy != null) {

            strategy!!.score()

            strategy!!.decision()


        }
    }

    fun setRegularization(rate: Float) {
        this.l2Lambda = rate
    }

    interface Strategy {

        fun decision()

        fun score()

    }

    fun setStrategy(strategy: Strategy) {
        this.strategy = strategy
    }

    fun clone(): NeuralNet {
        val clone = Util.setupDecisionNet(input!!.size, hiddenLayers!![0].size, if (hiddenLayers!!.size > 1) hiddenLayers!![1].size else 0, if (hiddenLayers!!.size > 2) hiddenLayers!![2].size else 0, output!!.size)
        Util.generateWeights(clone)
        for (i in 0 until hiddenLayers!![0].size) {
            clone.hiddenLayers!![0][i].weights = Util.copyWeights(hiddenLayers!![0][i].weights!!)
        }
        if (hiddenLayers!!.size > 1) {
            for (i in 0 until hiddenLayers!![1].size) {
                clone.hiddenLayers!![1][i].weights = Util.copyWeights(hiddenLayers!![1][i].weights!!)
            }
        }
        if (hiddenLayers!!.size > 2) {
            for (i in 0 until hiddenLayers!![2].size) {
                clone.hiddenLayers!![2][i].weights = Util.copyWeights(hiddenLayers!![2][i].weights!!)
            }
        }
        for (i in output!!.indices) {
            clone.output!![i].weights = Util.copyWeights(output!![i].weights!!)
        }
        clone.setSoftMax(softMax)
        return clone
    }


    /**
     * https://jamesmccaffrey.wordpress.com/2017/06/29/implementing-neural-network-l2-regularization/
     * @param l2
     */
    fun setL2(l2: Float) {
        this.l2Lambda = 1f - l2
    }

    fun setLearningRateDecay(decay: Float) {
        this.learningRateDecay = learningRateDecay
    }

    fun onEpochFinished() {
        if (learningRateDecay > 0f) {
            learningRate *= 1f - learningRateDecay
        }
    }

    /**
     * https://jamesmccaffrey.wordpress.com/2017/06/06/neural-network-momentum/
     * @param momentum
     */
    fun setMomentum(momentum: Float) {
        this.momentum = momentum
    }


}
