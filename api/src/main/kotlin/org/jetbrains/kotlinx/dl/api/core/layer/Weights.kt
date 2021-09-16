package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.extension.convertTensorToMultiDimArray

/**
 *  Layer's weights
 *  Require parent model to be set on layer
 */
public var Layer.weights: Map<String, Array<*>>
    get() {
        if (this !is ParametrizedLayer) return emptyMap()

        val model = parentModel
        requireNotNull(model) { "Layer '$name' is not related to any model" }

        val variablesOrder = variables
        val runner = model.session.runner()
        variablesOrder.map { it.variable }.forEach(runner::fetch)
        val weights = runner.run().map { it.convertTensorToMultiDimArray() }

        return variablesOrder.map { it.name }.zip(weights).toMap()
    }
    set(weights) {
        if (this !is ParametrizedLayer) return

        val model = parentModel
        requireNotNull(model) { "Layer '$name' is not related to any model" }

        for ((name, value) in weights) {
            model.assignVariable(name, value)
        }
    }