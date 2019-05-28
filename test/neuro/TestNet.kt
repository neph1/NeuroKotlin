import neuro.Util

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Rickard
 */
object TestNet {

    fun test() {
        val n = Util.setupDecisionNet(2, 2, 0, 0, 1)
        n.output!![0].weights!![0] = 0.3f
        n.output!![0].weights!![1] = 0.9f
        n.hiddenLayers!![0].get(0).weights!![0] = 0.1f
        n.hiddenLayers!![0][0].weights!![1] = 0.8f
        n.hiddenLayers!![0][1].weights!![0] = 0.4f
        n.hiddenLayers!![0][1].weights!![1] = 0.6f
        n.learningRate = 1f
        n.setInput(floatArrayOf(0.35f, 0.9f))

        n.feedForward()

        println("Output " + n.output!![0].getOutput())
        val error = n.getErrorDerivate(floatArrayOf(0.5f), true)

        println("Error " + error[0])
        println("Hidden output " + n.hiddenLayers!![0][0].output)

        println("New output weight 0 " + n.output!![0].weights!![0] + " Expected: 0.272392")
        println("New output weight 1 " + n.output!![0].weights!![1] + " Expected: 0.87305")

        println("New hidden weight 0 " + n.hiddenLayers!![0][0].weights!![0] + " Expected: 0.09916")
        println("New hidden weight 1 " + n.hiddenLayers!![0][0].weights!![1] + " Expected: 0.7978")
        println("New hidden weight 2 " + n.hiddenLayers!![0][1].weights!![0] + " Expected: 0.3972")
        println("New hidden weight 3 " + n.hiddenLayers!![0][1].weights!![1] + " Expected: 0.5928")

        n.setInput(floatArrayOf(0.35f, 0.9f))

        n.feedForward()

        println("Output " + n.output!![0].getOutput())
    }
}
