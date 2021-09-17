/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.TrainableModel
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Base abstract class for all layers.
 */
public sealed class Layer {
    /**
     * Layer name. Would be changed if empty during model compilation.
     */
    public abstract var name: String

    /** Output data tensor shape. */
    public lateinit var outputShape: TensorShape

    /** Model where this layer is used. */
    public var parentModel: TrainableModel? = null

    /** Returns number of input parameters. */
    protected var fanIn: Int = Int.MIN_VALUE

    /** Returns number of output parameters. */
    protected var fanOut: Int = Int.MIN_VALUE

    /** Returns inbound layers. */
    public var inboundLayers: MutableList<Layer> = mutableListOf()

    /** Returns outbound layers. */
    public var outboundLayers: MutableList<Layer> = mutableListOf()

    /** Important part of functional API. It takes [layers] as input and saves them to the [inboundLayers] of the given layer. */
    public operator fun invoke(vararg layers: Layer): Layer {
        inboundLayers = layers.toMutableList()
        return this
    }
}

public data class OperandWithShape(
    val operand: Operand<Float>,
    val shape: Shape
)

public abstract class NoInputsLayer : Layer() {
    /**
     * Builds main layer input transformation with [tf]. Depends on [Layer] type.
     *
     * @param [tf] TensorFlow graph API for building operations.
     */
    public abstract fun build(
        tf: Ops,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): OperandWithShape
}

public abstract class SingleInputLayer : Layer() {
    /**
     * Builds main layer input transformation with [tf]. Depends on [Layer] type.
     */
    public abstract fun build(
        tf: Ops,
        input: OperandWithShape,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): OperandWithShape
}

public abstract class MultipleInputsLayer : Layer() {
    /**
     * Builds main layer input transformation with [tf]. Depends on [Layer] type.
     */
    public abstract fun build(
        tf: Ops,
        input: List<OperandWithShape>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): OperandWithShape
}

internal fun requireArraySize(array: LongArray, size: Int, name: String) =
    require(array.size == size) {
        "$name is expected to have size equal $size but got ${array.size}"
    }
