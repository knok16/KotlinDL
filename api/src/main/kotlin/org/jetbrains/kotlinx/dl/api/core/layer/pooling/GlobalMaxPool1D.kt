/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.pooling

import org.jetbrains.kotlinx.dl.api.core.layer.OperandWithShape
import org.jetbrains.kotlinx.dl.api.core.layer.SingleInputLayer
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Global max pooling operation for 1D temporal data (e.g. audio, timeseries, etc.).
 *
 * Downsamples the input by taking the maximum value over time dimension.
 */
public class GlobalMaxPool1D(
    override var name: String = ""
) : SingleInputLayer() {
    private fun computeOutputShape(inputShape: Shape): Shape {
        return Shape.make(inputShape.size(0), inputShape.size(2))
    }

    override fun build(
        tf: Ops,
        input: OperandWithShape,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): OperandWithShape {
        return OperandWithShape(
            tf.max(input.operand, tf.constant(1)),
            computeOutputShape(input.shape)
        )
    }

    override fun toString(): String =
        "GlobalMaxPool1D(name=$name)"
}
