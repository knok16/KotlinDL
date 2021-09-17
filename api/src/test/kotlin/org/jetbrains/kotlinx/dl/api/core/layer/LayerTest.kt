package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.shape.*
import org.jetbrains.kotlinx.dl.api.core.util.DATA_PLACEHOLDER
import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.junit.jupiter.api.Assertions
import org.tensorflow.*
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder

enum class RunMode {
    EAGER,
    GRAPH
}

open class LayerTest {

    private fun getInputOp(tf: Ops, input: Array<*>): Operand<Float> {
        return when (input.shape.numDimensions()) {
            1 -> tf.constant(input.map { it as Float }.toFloatArray())
            2 -> tf.constant(input.cast2D<FloatArray>())
            3 -> tf.constant(input.cast3D<FloatArray>())
            4 -> tf.constant(input.cast4D<FloatArray>())
            5 -> tf.constant(input.cast5D<FloatArray>())
            else -> throw IllegalArgumentException("Inputs with more than 5 dimensions are not supported yet!")
        }
    }

    private fun getLayerOutputOp(
        tf: Ops,
        layer: SingleInputLayer,
        input: Array<*>,
    ): Output<*> {
        val inputShape = input.shape
        val inputOp = getInputOp(tf, input)
        val isTraining = tf.constant(true)
        val numberOfLosses = tf.constant(1.0f)
        return layer.build(tf, OperandWithShape(inputOp, inputShape), isTraining, numberOfLosses).operand.asOutput()
    }

    private fun runLayerInEagerMode(
        layer: SingleInputLayer,
        input: Array<*>,
    ): Tensor<*> {
        EagerSession.create().use {
            val tf = Ops.create()
            val outputOp = getLayerOutputOp(tf, layer, input)
            return outputOp.tensor()
        }
    }

    private fun runLayerInGraphMode(
        layer: SingleInputLayer,
        input: Array<*>,
    ): Tensor<*> {
        Graph().use { graph ->
            Session(graph).use { session ->
                val tf = Ops.create(graph)
                val outputOp = getLayerOutputOp(tf, layer, input)

                if (layer is ParametrizedLayer) layer.initialize(session)

                return session.runner().fetch(outputOp).run().first()
            }
        }
    }

    /**
     * Checks the output of a layer given the input data is equal to the expected output.
     *
     * This takes care of building and running the layer instance ([layer]), in either of
     * Eager or Graph mode execution ([runMode]) to verify the output of layer for the given
     * input data ([input]), is equal to the expected output ([expectedOutput]).
     *
     * Note that this method could be used for a layer with any input/output dimensionality.
     */
    protected fun assertLayerOutputIsCorrect(
        layer: SingleInputLayer,
        input: Array<*>,
        expectedOutput: Array<*>,
        runMode: RunMode = RunMode.EAGER,
    ) {
        val output = when (runMode) {
            RunMode.EAGER -> runLayerInEagerMode(layer, input)
            RunMode.GRAPH -> runLayerInGraphMode(layer, input)
        }
        output.use {
            val outputShape = shapeFromDims(*output.shape())
            val outputArray = getFloatArrayOfShape(outputShape).let {
                when (outputShape.numDimensions()) {
                    1 -> it as Array<Float>
                    2 -> it.cast2D<FloatArray>()
                    3 -> it.cast3D<FloatArray>()
                    4 -> it.cast4D<FloatArray>()
                    5 -> it.cast5D<FloatArray>()
                    else -> throw IllegalArgumentException("Arrays with more than 5 dimensions are not supported yet!")
                }
            }
            output.copyTo(outputArray)
            Assertions.assertArrayEquals(
                expectedOutput,
                outputArray
            )
        }
    }

    /**
     * Checks the computed output shape of layer is equal to the expected output shape.
     *
     * Essentially, this method invokes the `computeOutputShape` of a layer instance ([layer])
     * given an input shape array ([inputShapeArray]) and verifies its output is equal to the
     * expected output shape ([expectedOutputShape]).
     */
    protected fun assertLayerComputedOutputShape(
        layer: SingleInputLayer,
        inputShapeArray: LongArray,
        expectedOutputShape: LongArray,
    ) {
        val tf = Ops.create()
        val inputShape = shapeFromDims(*inputShapeArray)
        val input = tf.placeholder(getDType(), Placeholder.shape(inputShape))
        val training = tf.placeholder(Boolean::class.javaObjectType, Placeholder.shape(Shape.scalar()))
        val numberOfLossesOp = tf.placeholder(getDType(), Placeholder.shape(Shape.scalar()))

        val outputShape =
            layer.build(tf, OperandWithShape(input, inputShape), training, numberOfLossesOp).shape.toLongArray()
        Assertions.assertArrayEquals(
            expectedOutputShape,
            outputShape,
            "Computed output shape differs from expected output shape!",
        )
    }
}
