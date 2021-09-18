/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.visualization

import examples.inference.lenet5
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.summary.logSummary
import org.jetbrains.kotlinx.dl.dataset.mnist
import org.jetbrains.kotlinx.dl.visualization.letsplot.*
import org.jetbrains.kotlinx.dl.visualization.swing.*

private const val EPOCHS = 1
private const val TRAINING_BATCH_SIZE = 500
private const val TEST_BATCH_SIZE = 1000

/**
 * This examples demonstrates model activations and Conv2D filters visualisation.
 *
 * Model is trained on Mnist dataset.
 */
fun main() {

    val (train, test) = mnist()

    val sampleIndex = 42
    val x = test.getX(sampleIndex)
    val y = test.getY(sampleIndex).toInt()

    lenet5().use {

        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.logSummary()

        it.fit(
            dataset = train,
            validationRate = 0.1,
            epochs = EPOCHS,
            trainBatchSize = TRAINING_BATCH_SIZE,
            validationBatchSize = TEST_BATCH_SIZE
        )

        val numbersPlots = List(3) { imageIndex ->
            flattenImagePlot(imageIndex, test, it::predict)
        }
        columnPlot(numbersPlots, 3, 256).show()

        val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
        println("Accuracy $accuracy")

        val weights = it.weights
        val fstConv2D = weights["conv2d_1_conv2d_kernel"] as Array<*>
        val sndConv2D = weights["conv2d_3_conv2d_kernel"] as Array<*>

        // lets-plot approach
        filtersPlot(fstConv2D, columns = 16).show()
        filtersPlot(sndConv2D, columns = 16).show()

        // swing approach
        drawFilters(fstConv2D, colorCoefficient = 10.0)
        drawFilters(sndConv2D, colorCoefficient = 10.0)

        val layersActivations = modelActivationOnLayersPlot(it, x)
        val (prediction, activations) = it.predictAndGetActivations(x)
        println("Prediction: $prediction")
        println("Ground Truth: $y")

        // lets-plot approach
        layersActivations[0].show()
        layersActivations[1].show()

        // swing approach
        drawActivations(activations)
    }
}
