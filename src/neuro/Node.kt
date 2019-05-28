/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuro


import neuro.Util.sigmoid
import kotlin.math.tanh
import kotlin.random.Random

/**
 *
 * @author rickard
 */
class Node {
    var inConnections: Array<Node>? = null
    var outConnections: Array<Node>? = null
    var weights: FloatArray? = null
    internal var weightsPrime: FloatArray? = null
    //float[] weightsPrime;
    internal var input: Float = 0.toFloat()
    var output: Float = 0.toFloat()
        internal set
    var bias: Float = 0.toFloat()
    private val addInputNoise = false
    private var thresholdZero = true
    private var delta = 0f
    private var previousDelta = 0f

    var nodeType = Activation.LOGISTIC

    private val inputNoiseVariance = 0.05f

    enum class Activation {
        LOGISTIC,
        LINEAR,
        BINARY,
        TANH,
        IDENTITY;
    }

    constructor(activation: Activation) {
        this.nodeType = activation
    }

    constructor() {}

    /**
     * hid_to_output_weights_gradient =  hidden_layer_state * error_deriv';
     * output_bias_gradient = sum(error_deriv, 2);
     * back_propagated_deriv_1 = (hid_to_output_weights * error_deriv) ...
     * .* hidden_layer_state .* (1 - hidden_layer_state);
     * \delta_{o1} = -(target_{o1} - out_{o1}) * out_{o1}(1 - out_{o1})
     * @param learningRate
     * @param error
     * @param l2Lambda
     * @param momentum
     */
    fun updateWeights(learningRate: Float, error: Float, l2Lambda: Float, momentum: Float) {
        for (i in weights!!.indices) {
            if (l2Lambda > 0) {
                weights!![i] -= l2Lambda * weights!![i]
            }
            delta = learningRate * (this!!.inConnections!![i].output * error)
            weights!![i] += delta // w+(e*input)
            if (momentum > 0) {
                weights!![i] += momentum * previousDelta // w+(e*input)

                previousDelta = delta
            }
        }
    }

    fun updateOutput() {
        input = 0f
        // add input from all incoming nodes
        for (i in inConnections!!.indices) {
            input += inConnections!![i].output * weights!![i] + bias
        }
        // update output
        when (nodeType) {
            Activation.BINARY -> {
                if (input > 0) {
                    output = 1f
                } else {
                    output = 0f
                }
            }
            Activation.IDENTITY, Activation.LINEAR -> {
                output = input
            }
            Activation.TANH -> {
                output = tanh(input)
            }
            // logistic
            else -> {
                output = sigmoid(input.toDouble()).toFloat() //input / (1f + input * input);
                if (thresholdZero && output < 0) {
                    output = 0f
                }
            }
        }
    }

    fun setInput(input: Float) {
        this.input = input
    }

    fun generateWeights(random: Random, mean: Float) {

        weights = FloatArray(inConnections!!.size)
        weightsPrime = FloatArray(inConnections!!.size)
        //weightsPrime = new float[inConnections.length];
        for (i in weights!!.indices) {
            weights!![i] = random.nextFloat() * mean * 2f - mean
        }
    }

    fun getError(targetValue: Float): Float {
        when (nodeType) {
            Activation.BINARY -> {
                if (targetValue != output) {
                    for (i in weights!!.indices) {
                        weights!![i] += if (output == 0f) input else -input
                    }
                }
                return 0f
            }
            Activation.IDENTITY,
            Activation.LINEAR -> {
                return targetValue - output
                //                return (float) (Math.pow(targetValue - output, 2) * 0.5f);
            }
            else -> {
                return (targetValue - output) * (1f - output) * output // (t-o)*(1-o) * o;
            }
        }
    }

    fun getHiddenError(error: Float): Float {
        when (nodeType) {
            Activation.LINEAR -> {
                return error - output
            }
            else -> {
                return error * (1f - output) * output // (t-o)*(1-o) * o;
            }
        }

    }

    override fun toString(): String {
        var string = "{"
        for (f in weights!!) {
            string += f.toString() + "f, "
        }
        string += "}"
        return string
    }

    fun setActivation(costFunction: Activation) {
        this.nodeType = costFunction
    }

    fun setThresholdZero(threshold: Boolean) {
        this.thresholdZero = threshold
    }


    fun getOutput(): Float {
        return output
    }


}
