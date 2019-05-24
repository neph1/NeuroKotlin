/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuro

import kotlin.random.Random
import neuro.Node.Activation
import kotlin.math.exp

/**
 *
 * @author rickard
 */
object Util {


    val random = Random(17997821)

    /**
     * Sets up a fully connected net with one hidden layer
     * @param inputNodes
     * @param hiddenNodes1
     * @param hidden1Activation
     * @param outputNodes
     * @param outputActivation
     * @return
     */
    fun setupDecisionNet(inputNodes: Int, hiddenNodes1: Int, hidden1Activation: Activation, outputNodes: Int, outputActivation: Activation): NeuralNet {
        return setupDecisionNet(inputNodes, hiddenNodes1, hidden1Activation, 0, Activation.LOGISTIC, 0, Activation.LOGISTIC, outputNodes, outputActivation)
    }

    /**
     * Sets up a fully connected net with two hidden layers
     * @param inputNodes
     * @param hiddenNodes1
     * @param hidden1Activation
     * @param outputNodes
     * @param outputActivation
     * @return
     */
    fun setupDecisionNet(inputNodes: Int, hiddenNodes1: Int, hidden1Activation: Activation, hiddenNodes2: Int, hidden2Activation: Activation, outputNodes: Int, outputActivation: Activation): NeuralNet {
        return setupDecisionNet(inputNodes, hiddenNodes1, hidden1Activation, hiddenNodes2, hidden2Activation, 0, Activation.LOGISTIC, outputNodes, outputActivation)
    }

    fun setupDecisionNet(inputNodes: Int, hiddenNodes1: Int, hiddenNodes2: Int, hiddenNodes3: Int, outputNodes: Int): NeuralNet {
        return setupDecisionNet(inputNodes, hiddenNodes1, Activation.LOGISTIC, hiddenNodes2, Activation.LOGISTIC, hiddenNodes3, Activation.LOGISTIC, outputNodes, Activation.LOGISTIC)
    }


    fun setupDecisionNet(inputNodes: Int, hiddenNodes1: Int, hidden1Activation: Activation, hiddenNodes2: Int, hidden2Activation: Activation, hiddenNodes3: Int, hidden3Activation: Activation, outputNodes: Int, outputActivation: Activation): NeuralNet {
        val decisionNet = NeuralNet()

        val hidden = Array<Node>(hiddenNodes1){Node(hidden1Activation)}
        for (i in 0 until hiddenNodes1) {
            hidden[i].inConnections = Array<Node>(inputNodes) {Node()}//arrayOfNulls<Node>(inputNodes)
            hidden[i].outConnections = Array<Node>(if (hiddenNodes2 > 0) hiddenNodes2 else outputNodes){Node()}
        }
        if (hiddenNodes2 > 0) {
            var hidden2 = Array<Node>(hiddenNodes2){Node(hidden2Activation)}
            for (i in 0 until hiddenNodes2) {
                hidden2[i].inConnections = Array<Node>(hiddenNodes1){Node()}
                hidden2[i].outConnections = Array<Node>(if (hiddenNodes3 > 0) hiddenNodes3 else outputNodes){Node()}

                for (j in 0 until hiddenNodes1) {
                    hidden2[i].inConnections!![j] = hidden[j]
                    hidden[j].outConnections!![i] = hidden2[i]
                }

            }
            if (hiddenNodes3 > 0) {
                var hidden3 = Array<Node>(hiddenNodes3){Node(hidden3Activation)}
                for (i in 0 until hiddenNodes3) {
                    hidden3[i].outConnections = Array<Node>(outputNodes){Node()}
                    hidden3[i].inConnections = Array<Node>(hiddenNodes2){Node()}

                    for (j in 0 until hiddenNodes2) {
                        hidden3[i].inConnections!![j] = hidden2[j]
                        hidden2[j].outConnections!![i] = hidden3[i]
                    }
                }
                decisionNet.hiddenLayers = arrayOf<Array<Node>>(hidden, hidden2, hidden3)
            } else {
                decisionNet.hiddenLayers = arrayOf<Array<Node>>(hidden, hidden2)
            }
        } else {
            decisionNet.hiddenLayers = arrayOf<Array<Node>>(hidden)
        }

        // output nodes

        val hiddenLayers = decisionNet.hiddenLayers!!.size
        val lastHiddenLayer = decisionNet.hiddenLayers!![hiddenLayers - 1].size
        decisionNet.output = Array<Node>(outputNodes){Node(outputActivation)}
        for (i in 0 until outputNodes) {
            decisionNet.output!![i].inConnections = Array<Node>(lastHiddenLayer){Node()}
        }
        for (i in 0 until lastHiddenLayer) {
            for (j in 0 until outputNodes) {
                decisionNet.output!![j].inConnections!![i] = decisionNet.hiddenLayers!![hiddenLayers - 1][i]
                decisionNet.hiddenLayers!![hiddenLayers - 1][i].outConnections!![j] = decisionNet.output!![j]
            }
        }

        // input nodes

        decisionNet.input = Array<Node>(inputNodes){Node()}
        for (i in 0 until inputNodes) {
            decisionNet.input!![i].outConnections = Array<Node>(hiddenNodes1){Node()}
            for (j in 0 until hiddenNodes1) {
                decisionNet.hiddenLayers!![0][j].inConnections!![i] = decisionNet.input!![i]
                decisionNet.input!![i].outConnections!![j] = decisionNet.hiddenLayers!![0][j]
            }
        }

        generateWeights(decisionNet)

        return decisionNet
    }

    private fun setupLayerOne() {

    }

    fun generateWeights(decisionNet: NeuralNet) {
        for (i in 0 until decisionNet.hiddenLayers!![0].size) {
            decisionNet.hiddenLayers!![0][i].generateWeights(random, 1f)
        }
        if (decisionNet.hiddenLayers!!.size > 1 && decisionNet.hiddenLayers!![1] != null) {
            for (i in 0 until decisionNet.hiddenLayers!![1].size) {
                decisionNet.hiddenLayers!![1][i].generateWeights(random, 1f)
            }
            if (decisionNet.hiddenLayers!!.size > 2) {
                for (i in 0 until decisionNet.hiddenLayers!![2].size) {
                    decisionNet.hiddenLayers!![2][i].generateWeights(random, 1f)
                }
            }
        }

        for (i in decisionNet.output!!.indices) {
            decisionNet.output!![i].generateWeights(random, 1f)
        }
    }

    fun sigmoid(x: Double): Double {
        //        return (1/( 1 + Math.pow(Math.E,(-1*x))));
        return 1 / (1 + exp(-x))
    }

    //    public static double log_sum_over_rows(double[] array){
    //
    //    }

    fun softMax(output: FloatArray) {
        val softMax = FloatArray(output.size)
        val max = output.max()
        val sum = output.sum()

    }

    fun randomDistributionSelection(softMaxOutput: FloatArray, randomValue: Float): Int {
        var softMaxSum = softMaxOutput[0]
        for (i in 0 until softMaxOutput.size - 1) {
            if (randomValue > softMaxSum) {
                softMaxSum += softMaxOutput[i + 1]
            } else {
                return i
            }
        }
        return softMaxOutput.size - 1
    }

    fun copyWeights(array: FloatArray): FloatArray {
        val copy = FloatArray(array.size)
        for (i in 0 until array.size) {
            copy[i]= array[i]
        }
        return copy
    }
}
