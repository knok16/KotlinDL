/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.jetbrains.kotlinx.dl.api.core.layer.OperandWithShape
import org.jetbrains.kotlinx.dl.api.core.layer.SingleInputLayer
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Layer that repeats the input [n] times.
 *
 * Input shape: `2D tensor of shape (num_samples, features)`.
 *
 * Output shape: `3D tensor of shape (num_samples, n, features)`.
 *
 * @property n Repetition factor.
 * @property [name] Custom layer name.
 * @constructor Creates [RepeatVector] object.
 *
 * @since 0.3
 */
public class RepeatVector(
    public val n: Int,
    override var name: String = ""
) : SingleInputLayer() {

    init {
        require(n >= 1) { "Number of repetitions (n) in RepeatVector should be positive but got $n" }
    }

    private fun computeOutputShape(inputShape: Shape): Shape {
        require(inputShape.numDimensions() == 2) {
            "Input tensor must have 2 dimensions but got ${inputShape.numDimensions()}"
        }
        return Shape.make(inputShape.size(0), n.toLong(), inputShape.size(1))
    }

    override fun build(
        tf: Ops,
        input: OperandWithShape,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): OperandWithShape {
        val x = tf.expandDims(input.operand, tf.constant(1))
        val pattern = tf.stack(listOf(tf.constant(1), tf.constant(n), tf.constant(1)))
        return OperandWithShape(
            tf.tile(x, pattern),
            computeOutputShape(input.shape)
        )
    }

    override fun toString(): String {
        return "RepeatVector"
    }
}
