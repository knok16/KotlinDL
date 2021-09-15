package org.jetbrains.kotlinx.dl.api.core.layer

import org.tensorflow.op.core.Variable

public interface TrainableLayer : ParametrizedLayer {
    /**
     * True, if layer's weights could be changed during training.
     * If false, layer's weights are frozen and could not be changed during the training.
     */
    public var isTrainable: Boolean
}

/**
 * Returns amount of parameters
 */
public val Layer.isTrainable: Boolean
    get() = if (this is TrainableLayer) isTrainable else false

/**
 * Returns a list of non-trainable, 'frozen' variables used in layers.
 */
public fun List<Layer>.frozenLayerVariables(): List<Variable<Float>> =
    filterIsInstance<ParametrizedLayer>()
        .filter { it !is TrainableLayer || !it.isTrainable }
        .flatMap { it.variables }
        .map { it.variable }

/**
 * Returns a list of trainable variables used in layers.
 */
public fun List<Layer>.trainableLayerVariables(): List<VariableDto> =
    filterIsInstance<TrainableLayer>().filter { it.isTrainable }.flatMap { it.variables }