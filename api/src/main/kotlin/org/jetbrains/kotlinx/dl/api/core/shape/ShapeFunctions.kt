/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.shape

import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Creates shape [Shape] from [LongArray].
 */
public fun shape(dims: LongArray?): Shape = when {
    dims == null -> Shape.unknown()
    dims.isEmpty() -> Shape.scalar()
    else -> Shape.make(dims[0], *dims.copyOfRange(1, dims.size))
}

/**
 * Creates shape [Operand] from [Shape].
 */
internal fun shapeOperand(tf: Ops, shape: Shape): Operand<Int> =
    tf.constant(shape.toIntArray())

/**
 * Extracts dimensions as [IntArray] from [Shape].
 */
internal fun Shape.toIntArray(): IntArray =
    IntArray(numDimensions()) { size(it).toInt() }

/**
 * Extracts dimensions as [LongArray] from [Shape].
 */
public fun Shape.toLongArray(): LongArray =
    LongArray(numDimensions()) { size(it) }

/**
 * Returns the value of a dimension
 *
 * @param i The index at which to retrieve a dimension.
 * @return The size of dimension i
 */
public operator fun Shape.get(i: Int): Long =
    size(i)

/**
 * Returns amount of elements in Tensor with the given shape.
 * Negative dimensions are ignored.
 */
public fun Shape.numElements(): Long =
    (0 until numDimensions()).asSequence().map(::size).filter { it >= 0 }.fold(1L, Long::times)

/**
 * Test whether dimension i in this shape is known
 *
 * @param [i] Target dimension to test
 * @return Whether dimension [i] is unknown (equal to -1)
 */
private fun Shape.isKnown(i: Int): Boolean {
    return this[i] != -1L
}

/**
 * Throw an exception if dimension [i] is unknown.
 *
 * @param [i] Target dimension to test
 * @throws IllegalStateException if dimension [i] is unknown
 */
public fun Shape.assertKnown(i: Int): Unit =
    check(isKnown(i)) { "Dimension $i in shape needs to be known." }

/**
 * Returns the head dimension.
 */
public val Shape.head: Long
    get() = get(0)

/**
 * Returns the tail dimensions.
 */
public val Shape.tail: LongArray
    get() = toLongArray().copyOfRange(1, numDimensions())

public fun Shape.isCompatible(shape: Shape): Boolean =
    this.numDimensions() == shape.numDimensions() && (0 until this.numDimensions()).all { this[it] == shape[it] }

/** Reshapes 2D array of floats to 1D array of floats. */
public fun reshape2DTo1D(dst: Array<FloatArray>, size: Int): FloatArray {
    val result = FloatArray(size) { 0.0f }

    var pos = 0

    for (i in dst.indices) {
        for (j in dst[i].indices) {
            result[pos] = dst[i][j]
            pos++
        }
    }

    return result
}

/** Reshapes 3D array of floats to 1D array of floats. */
public fun reshape3DTo1D(dst: Array<Array<FloatArray>>, size: Int): FloatArray {
    val result = FloatArray(size) { 0.0f }

    var pos = 0
    for (i in dst.indices) {
        for (j in dst[i].indices) {
            for (k in dst[i][j].indices) {
                result[pos] = dst[i][j][k]
                pos++
            }

        }
    }
    return result
}

/** Reshapes 4D array of floats to 1D array of floats. */
public fun reshape4DTo1D(dst: Array<Array<Array<FloatArray>>>, size: Int): FloatArray {
    val result = FloatArray(size) { 0.0f }

    var pos = 0
    for (i in dst.indices) {
        for (j in dst[i].indices) {
            for (k in dst[i][j].indices) {
                for (m in dst[i][j][k].indices) {
                    result[pos] = dst[i][j][k][m]
                    pos++
                }
            }
        }
    }
    return result
}

/**
 * Get shape of array of arrays (of arrays...) of Array of elems of any type.
 * If the most inner array does not have any elements its size is missed in result */
internal fun getShapeOfArray(data: Array<*>): Shape {
    fun appendPrimitiveArraySize(size: Int, acc: MutableList<Long>): LongArray {
        acc += size.toLong()
        return acc.toLongArray()
    }

    tailrec fun collectDims(data: Array<*>, acc: MutableList<Long>): LongArray {
        val firstElem = data[0] ?: return acc.toLongArray()
        acc += data.size.toLong()
        return when (firstElem) {
            is Array<*> -> collectDims(firstElem, acc)
            is BooleanArray -> appendPrimitiveArraySize(firstElem.size, acc)
            is ByteArray -> appendPrimitiveArraySize(firstElem.size, acc)
            is CharArray -> appendPrimitiveArraySize(firstElem.size, acc)
            is ShortArray -> appendPrimitiveArraySize(firstElem.size, acc)
            is IntArray -> appendPrimitiveArraySize(firstElem.size, acc)
            is LongArray -> appendPrimitiveArraySize(firstElem.size, acc)
            is FloatArray -> appendPrimitiveArraySize(firstElem.size, acc)
            is DoubleArray -> appendPrimitiveArraySize(firstElem.size, acc)
            else -> acc.toLongArray()
        }
    }
    return shape(collectDims(data, mutableListOf()))
}

/** Shape property of standard JVM array for better readability of code */
internal val Array<*>.shape: Shape get() = getShapeOfArray(this)

/**
 * Create an array of arrays (of arrays...) of Floats with specified [shape] and
 * initialized with given [initValue]. When the number of dimensions in result tensor
 * is bigger than 1, the last dimension array is FloatArray (instead of Array<Float>).
 */
internal fun getFloatArrayOfShape(shape: Shape, initValue: Float = 0.0f): Array<*> {
    fun getFloatArrayOfShape(shape: Shape, dimIndex: Int): Any = if (shape.numDimensions() - 1 == dimIndex) {
        FloatArray(shape.size(dimIndex).toInt()) { initValue }
    } else {
        Array(shape.size(dimIndex).toInt()) { getFloatArrayOfShape(shape, dimIndex + 1) }
    }
    return if (shape.numDimensions() == 1) {
        Array(shape.size(0).toInt()) { initValue }
    } else {
        getFloatArrayOfShape(shape, 0) as Array<*>
    }
}

internal fun Any?.castArray(): Array<*> = this as Array<*>

/** Cast Array<*> to Array<T> when sure about its dimensions where usually T is [FloatArray] */
internal inline fun <reified T> Array<*>.cast2D(): Array<T> =
    this.map { it as T }.toTypedArray()

/** Cast Array<*> to Array<Array<T>> when sure about its dimensions where usually T is [FloatArray] */
internal inline fun <reified T> Array<*>.cast3D(): Array<Array<T>> =
    this.map { it.castArray().cast2D<T>() }.toTypedArray()

/** Cast Array<*> to Array<Array<Array<T>>> when sure about its dimensions where usually T is [FloatArray] */
internal inline fun <reified T> Array<*>.cast4D(): Array<Array<Array<T>>> =
    this.map { it.castArray().cast3D<T>() }.toTypedArray()

/** Cast Array<*> to Array<Array<Array<Array<T>>>> when sure about its dimensions where usually T is [FloatArray] */
internal inline fun <reified T> Array<*>.cast5D(): Array<Array<Array<Array<T>>>> =
    this.map { it.castArray().cast4D<T>() }.toTypedArray()
