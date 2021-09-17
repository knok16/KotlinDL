/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.activation

import org.jetbrains.kotlinx.dl.api.core.layer.LayerWithActivation
import org.jetbrains.kotlinx.dl.api.core.layer.OperandWithShape
import org.jetbrains.kotlinx.dl.api.core.layer.SingleInputLayer
import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Base class for all layer class representing activation functions.
 *
 * By default, it is marked as __not trainable__ layer with no extra
 * parameters and weights but having the activation on it.
 *
 * By default, it defines returning the output with the same shape
 * as the input Operand.
 */
public abstract class AbstractActivationLayer : SingleInputLayer(), LayerWithActivation {
    /**
     * Applies the activation functions to the [input] to produce the output.
     *
     * @param [tf] TensorFlow graph API for building operations.
     * @param [input] TensorFlow graph leaf node representing layer output before activation function.
     */
    protected abstract fun forward(
        tf: Ops,
        input: Operand<Float>
    ): Operand<Float>

    override fun build(
        tf: Ops,
        input: OperandWithShape,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): OperandWithShape = OperandWithShape(
        forward(tf, input.operand),
        input.shape
    )
}
