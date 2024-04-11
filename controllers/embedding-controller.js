
const tf = require('@tensorflow/tfjs-node');

const fs = require("fs");
const modelPath = '/Users/mma/git/tf-ai/embedding';

const docs = ['Well done!',
    'Good work',
    'Great effort',
    'nice work',
    'Excellent!',
    'Weak',
    'Poor effort!',
    'not good',
    'poor work',
    'Could have done better.']
const labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
//TODO: convert text to embedding
const test = [[6, 2, 0, 0],
[3, 1, 0, 0],
[7, 4, 0, 0],
[8, 1, 0, 0],
[9, 0, 0, 0],
[10, 0, 0, 0],
[5, 4, 0, 0],
[11, 3, 0, 0],
[5, 1, 0, 0],
[12, 13, 2, 14]];

const encodedDocs = tf.oneHot(tf.tensor1d(docs, 'int32'), docs.length).dataSync();

const embedding = (async (req, res) => {
    console.log("Load an existing model");
    let model = await load();
    if (!model) {
        console.log("model not found, train a new one");
        model = await train();
    }
    if (model) {
        const result = await getResult(model);
        console.log("predit result", result);
        res.send(result);
    }
})

async function train() {

    const model = tf.sequential();
    model.add(tf.layers.embedding({
        inputDim: 100,//vocabularySize,
        outputDim: 8,//embeddingSize,
        inputLength: 4,//maxLen
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
    await model.fit(tf.tensor2d(test), tf.tensor1d(labels), {
        epochs: 500,
        callbacks: {
            //onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
        }
    });

    console.log('Evaluating model...');
    const [testLoss, testAcc] =
        await model.evaluate(tf.tensor2d(test), tf.tensor1d(labels), { batchSize: 100 });
    console.log(`Evaluation loss: ${(await testLoss.data())[0].toFixed(4)}`);
    console.log(`Evaluation accuracy: ${(await testAcc.data())[0].toFixed(4)}`);

    await model.save('file://' + modelPath);
    return model;
}
async function getResult(model) {
    const arr = [];
    for (let x = 0; x < 10; x++) {
        arr.push({ x: test[x] });
        let result = await model.predict(tf.tensor([test[x]]));
        console.log(result);
        arr[x].y = Number(await result.data());
    }
    return arr;
}
async function load() {
    if (fs.existsSync(modelPath + '/model.json')) {
        return await tf.loadLayersModel('file://' + modelPath + '/model.json');
    }
    return null;
}
module.exports = {
    embedding
}