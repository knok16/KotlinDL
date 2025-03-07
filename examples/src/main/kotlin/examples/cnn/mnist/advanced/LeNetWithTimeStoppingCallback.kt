/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.cnn.mnist.advanced

import examples.cnn.models.buildLetNet5Classic
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.callback.TimeStopping
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.optimizer.ClipGradientByValue
import org.jetbrains.kotlinx.dl.api.core.summary.logSummary
import org.jetbrains.kotlinx.dl.dataset.handler.NUMBER_OF_CLASSES
import org.jetbrains.kotlinx.dl.dataset.mnist

private const val EPOCHS = 10
private const val TRAINING_BATCH_SIZE = 1000
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L
private const val SEED = 12L
private const val TEST_BATCH_SIZE = 1000

private val lenet5Classic = buildLetNet5Classic(
    image_width = IMAGE_SIZE,
    image_height = IMAGE_SIZE,
    num_channels = NUM_CHANNELS,
    num_classes = NUMBER_OF_CLASSES,
    layers_activation = Activations.Tanh,
    classifier_activation = Activations.Linear,
    random_seed = SEED,
)

/**
 * This example shows how to do image classification from scratch using [lenet5Classic], without leveraging pre-trained weights or a pre-made model.
 * We demonstrate the workflow on the Mnist classification dataset.
 *
 * It includes:
 * - dataset loading from S3
 * - callback definition
 * - model compilation with [TimeStopping] callback
 * - model summary
 * - model training
 * - model evaluation
 */
fun lenetWithTimeStoppingCallback() {
    val (train, test) = mnist()

    lenet5Classic.use {
        val timeStopping = TimeStopping(seconds = 30, verbose = true)
        it.compile(
            optimizer = Adam(clipGradient = ClipGradientByValue(0.1f)),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY,
            callback = timeStopping,
        )

        it.logSummary()

        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)

        val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy: $accuracy")
    }
}

fun main(): Unit = lenetWithTimeStoppingCallback()
