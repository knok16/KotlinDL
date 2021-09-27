/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.merge

import org.jetbrains.kotlinx.dl.api.core.layer.NoGradients
import org.jetbrains.kotlinx.dl.api.core.shape.get
import org.jetbrains.kotlinx.dl.api.core.shape.shape
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Layer that concatenates a list of inputs.
 *
 * It takes as input a list of tensors, all the same shape except
 * for the concatenation axis, and returns a single tensor that is the concatenation of all inputs.
 *
 * @property [axis] Axis along which to concatenate.
 */
public class Concatenate(
    public var axis: Int,
    name: String = ""
) : AbstractMerge("ConcatenateLayer", name), NoGradients {
    init {
        isTrainable = false
    }

    override fun computeOutputShapeFromInboundLayers(): Shape {
        val inputShapes = inboundLayers.map { it.outputShape }

        val firstShape = inputShapes[0]

        val axisToConcatenateAlong = if (axis == -1) { // it influences on nasmobilemodel
            firstShape.numDimensions() - 1 // concatenate along last axis if axis is not specified
        } else {
            axis
        }

        outputShape = shape(LongArray(firstShape.numDimensions()) { i ->
            if (i == axisToConcatenateAlong) {
                inputShapes.sumOf { it[axisToConcatenateAlong] } // concatenated dimension
            } else {
                firstShape[i]
            }
        })
        return outputShape
    }

    override fun checkInputShapesOfInputOperands(input: List<Operand<Float>>) {
        require(input.size > 1) { "The number of input layers should be more than 1." }

        val firstInputShape = input[0].asOutput().shape()

        for (layer in input) {
            val tensorShape = layer.asOutput().shape()
            require(
                (0 until firstInputShape.numDimensions())
                    .asSequence()
                    .filterNot { it == axis }
                    .all { firstInputShape[it] == tensorShape[it] }
            ) {
                "A Concatenate layer requires inputs with matching shapes except for the concat axis. " +
                        "But shapes are the following: shape of first input is $firstInputShape and shape of layer $layer is $tensorShape."
            }
        }
    }

    override fun mergeFunction(input: List<Operand<Float>>, tf: Ops): Operand<Float> {
        return tf.concat(input, tf.constant(axis))
    }
}
