/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.cv

import examples.onnx.cv.custom.efficientnet.*
import examples.onnx.cv.resnet.resnet18easyPrediction
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.junit.jupiter.api.Test

class OnnxEfficientNetTestSuite {
    @Test
    fun efficientNetB0PredictionTest() {
        efficientNetB0Prediction()
    }

    @Test
    fun efficientNetB0AdditionalTrainingTest() {
        efficientNetB0AdditionalTraining()
    }

    @Test
    fun efficientNetB0EasyPredictionTest() {
        efficientNetB0EasyPrediction()
    }

    @Test
    fun efficientNetB1PredictionTest() {
        runONNXImageRecognitionPrediction(ONNXModels.CV.EfficientNetB1, resizeTo = Pair(240, 240))
    }

    @Test
    fun efficientNetB1AdditionalTrainingTest() {
        runONNXAdditionalTraining(ONNXModels.CV.EfficientNetB1noTop, resizeTo = Pair(240, 240))
    }

    @Test
    fun efficientNetB2PredictionTest() {
        runONNXImageRecognitionPrediction(ONNXModels.CV.EfficientNetB2, resizeTo = Pair(260, 260))
    }

    @Test
    fun efficientNetB2AdditionalTrainingTest() {
        runONNXAdditionalTraining(ONNXModels.CV.EfficientNetB2noTop, resizeTo = Pair(260, 260))
    }

    @Test
    fun efficientNetB3PredictionTest() {
        runONNXImageRecognitionPrediction(ONNXModels.CV.EfficientNetB3, resizeTo = Pair(300, 300))
    }

    @Test
    fun efficientNetB3AdditionalTrainingTest() {
        runONNXAdditionalTraining(ONNXModels.CV.EfficientNetB3noTop, resizeTo = Pair(300, 300))
    }

    @Test
    fun efficientNetB4PredictionTest() {
        runONNXImageRecognitionPrediction(ONNXModels.CV.EfficientNetB4, resizeTo = Pair(380, 380))
    }

    @Test
    fun efficientNetB4AdditionalTrainingTest() {
        runONNXAdditionalTraining(ONNXModels.CV.EfficientNetB4noTop, resizeTo = Pair(380, 380))
    }

    @Test
    fun efficientNetB5PredictionTest() {
        runONNXImageRecognitionPrediction(ONNXModels.CV.EfficientNetB5, resizeTo = Pair(456, 456))
    }

    @Test
    fun efficientNetB5AdditionalTrainingTest() {
        runONNXAdditionalTraining(ONNXModels.CV.EfficientNetB5noTop, resizeTo = Pair(456, 456))
    }

    @Test
    fun efficientNetB6PredictionTest() {
        runONNXImageRecognitionPrediction(ONNXModels.CV.EfficientNetB6, resizeTo = Pair(528, 528))
    }

    @Test
    fun efficientNetB6AdditionalTrainingTest() {
        runONNXAdditionalTraining(ONNXModels.CV.EfficientNetB6noTop, resizeTo = Pair(528, 528))
    }

    @Test
    fun efficientNetB7PredictionTest() {
        runONNXImageRecognitionPrediction(ONNXModels.CV.EfficientNetB7, resizeTo = Pair(600, 600))
    }

    @Test
    fun efficientNetB7AdditionalTrainingTest() {
        runONNXAdditionalTraining(ONNXModels.CV.EfficientNetB7noTop, resizeTo = Pair(600, 600))
    }

    @Test
    fun efficientNetB7EasyPredictionTest() {
        efficientNetB7EasyPrediction()
    }
}


