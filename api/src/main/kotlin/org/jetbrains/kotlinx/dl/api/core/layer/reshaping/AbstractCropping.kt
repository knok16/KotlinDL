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
 * Abstract Cropping layer used as the base layer for all the cropping layers.
 *
 * @property [croppingInternal] Cropping size values; currently, they are not used in the implementation
 * of this abstract class and each subclassed layer uses its own copy of the cropping size values.
 */
public abstract class AbstractCropping(
    public val croppingInternal: Array<IntArray>
) : SingleInputLayer() {
    override fun build(
        tf: Ops,
        input: OperandWithShape,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): OperandWithShape = OperandWithShape(
        crop(tf, input.operand),
        computeOutputShape(input.shape)
    )

    protected abstract fun computeOutputShape(inputShape: Shape): Shape

    /**
     * The actual implementation of cropping operation which each subclassed layer needs to
     * implement. This method will then be called from [build] method to crop the input tensor.
     */
    protected abstract fun crop(tf: Ops, input: Operand<Float>): Operand<Float>
}
