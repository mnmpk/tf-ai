
const tf = require('@tensorflow/tfjs-node');
const fs = require("fs");
const modelPath = './transformer/model';
const {Transformer} = require('../transformer/model');


const call = (async (req, res) => {

    // Define the data
    const xTrain = tf.tensor2d([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
    const yTrain = tf.oneHot(tf.tensor1d([0, 1, 2], 'int32'), 3);
    const xTest = tf.tensor2d([[13, 14, 15, 16], [17, 18, 19, 20]]);

    // Define hyperparameters
    const vocabSize = 21;
    const numClasses = 3;
    const numLayers = 2;
    const numHeads = 4;
    const hiddenDim = 32;
    const dropoutRate = 0.1;

    // Create and compile the model
    const transformer = new Transformer(vocabSize, numClasses, numLayers, numHeads, hiddenDim, dropoutRate);
    transformer.summary();
    const optimizer = tf.train.adam();
    transformer.model.compile({ optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    // Train the model
    transformer.model.fit(xTrain, yTrain, { epochs: 10, batchSize: 3 }).then(() => {
        // Perform inference
        const predictions = transformer.model.predict(xTest);
        predictions.print();
    });

    res.send("OK");
})


module.exports = {
    call
}