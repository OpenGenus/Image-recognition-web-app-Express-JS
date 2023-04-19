const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const classes = require('./class_name');


const TOP_K = 1; // Number of top predictions to return
async function predictImageContents(imageData) {
    console.log("AT TENSORFLOW --->", imageData);

    const model = await tf.loadLayersModel(
        "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json"
    );
    // Load the class names
    //const classes = await fetch(CLASSES_URL).then((response) => response.json());

    // Preprocess the image
    const uint8Array = new Uint8Array(imageData.buffer);
    const decodedImage = tf.node.decodeImage(uint8Array);
    const resizedImage = tf.image.resizeBilinear(decodedImage, [224, 224]);
    const normalizedImage = resizedImage.div(255.0).expandDims();

    // Make the prediction
    const prediction = model.predict(normalizedImage);

    // Get the top k predictions
    const topK = await prediction.topk(TOP_K);

    // Get the predicted class and label
    const classIndex = topK.indices.dataSync()[0];
    const label = classes[classIndex];

    // Return the result
    return { classIndex, label };
}

module.exports = predictImageContents;
