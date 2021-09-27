/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.shape.get
import org.jetbrains.kotlinx.dl.api.core.shape.head
import org.jetbrains.kotlinx.dl.api.core.shape.numElements
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Constant

/**
 * Flattens the input. Does not affect the batch size.
 *
 * @property [name] Custom layer name.
 * @constructor Creates [Flatten] object.
 */
public class Flatten(name: String = "") : Layer(name) {
    private lateinit var units: Constant<Int>

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        val amountOfNeuronsInFlattenLayer = (1 until inputShape.numDimensions())
            .asSequence().map { inputShape[it].toInt() }.fold(1, Int::times)
        units = tf.constant(intArrayOf(-1, amountOfNeuronsInFlattenLayer))

        fanIn = inputShape.numElements().toInt()
        fanOut = amountOfNeuronsInFlattenLayer
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        // leaves unknown dimensions unknown
        return Shape.make(inputShape.head, inputShape.numElements())
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return tf.reshape(input, units)
    }

    override var weights: Map<String, Array<*>>
        get() = emptyMap()
        set(value) = assignWeights(value)

    override val hasActivation: Boolean get() = false

    override val paramCount: Int get() = 0

    override fun toString(): String {
        return "Flatten"
    }
}
