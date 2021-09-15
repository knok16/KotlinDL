package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.shape.numElements
import org.tensorflow.Session
import org.tensorflow.op.core.Variable

public interface ParametrizedLayer {
    /**
     * Layer's variables
     */
    public val variables: List<VariableDto>

    /**
     * Returns amount of parameters
     */
    public val ParametrizedLayer.paramCount: Int
        get() = variables.sumOf { it.shape.numElements() }.toInt()
}

/**
 * Returns amount of parameters
 */
public val Layer.paramCount: Int
    get() = if (this is ParametrizedLayer) paramCount else 0

/**
 * Returns all variables used in all layers
 */
public fun List<Layer>.layersVariables(): List<Variable<Float>> =
    filterIsInstance<ParametrizedLayer>().flatMap { it.variables }.map { it.variable }

public fun ParametrizedLayer.initialize(session: Session) {
    // Only run the session if there is initializer ops.
    if (variables.isNotEmpty()) {
        val runner = session.runner()
        variables.map { it.initOp }.forEach(runner::addTarget)
        runner.run()
    }
}