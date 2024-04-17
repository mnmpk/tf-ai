
const tf = require('@tensorflow/tfjs-node');
const fs = require("fs");
const modelPath = './prediction';

const map = [
    [1,1,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,0,0,0,1,1],
    [0,1,0,0,1,1,1,1,1,0],
    [1,1,0,0,0,0,0,1,0,0],
    [1,0,0,0,0,0,1,1,0,0],
    [1,1,1,1,1,1,1,0,0,0],
    [0,0,0,1,0,1,0,0,0,0],
    [0,0,1,1,0,1,1,0,0,0],
    [0,0,1,0,0,0,1,1,0,0],
    [0,0,1,0,0,0,0,1,1,1],
];
const routes = [
    [[0,0],[0,1],[1,1],[2,1],[3,1],[4,1],[0,0]]
]

const predict = (async (req, res) => {
    console.log("Load an existing model");
    let model = await load();
    if (!model) {
        console.log("model not found, train a new one");
        model = await train();
    }
    if (model) {
        let result = await model.predict();
        res.send(result);
    }
})

async function train() {
    const model = tf.sequential();
    model.add(tf.layers.embedding({
        inputDim: vocabulary.length,//vocabularySize,
        outputDim: 32,//embeddingSize,
        inputLength: maxLen,//maxLen
    }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    model.compile({
        loss: 'binaryCrossentropy',
        optimizer: 'adam',
        metrics: ['acc']
    });
    model.summary();

    // Train the model.
    await model.fit(tf.tensor2d(encodedDocs), tf.tensor1d(labels), {
        epochs: 500,
        callbacks: {
            //onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
        }
    });

    console.log('Evaluating model...');
    const [testLoss, testAcc] =
        await model.evaluate(tf.tensor2d(encodedDocs), tf.tensor1d(labels), { batchSize: 100 });
    console.log(`Evaluation loss: ${(await testLoss.data())[0].toFixed(4)}`);
    console.log(`Evaluation accuracy: ${(await testAcc.data())[0].toFixed(4)}`);

    await model.save('file://' + modelPath);
    return model;
}
module.exports = {
    predict
}