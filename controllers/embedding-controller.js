
const tf = require('@tensorflow/tfjs-node');
const fs = require("fs");
const modelPath = './embedding';

const maxLen = 50;
const vocabulary = ['', 'well', 'done', 'good', 'work', 'great', 'effort', 'nice', 'work', 'excellent', 'weak', 'bad', 'poor', 'effort', 'not', 'work', 'could', 'have', 'done', 'better'];
const docs = ['',
    'good', 'great', 'nice',
    'Well done!',
    'Good work',
    'Great effort',
    'nice work',
    'Excellent!',
    'Weak',
    'Poor effort!',
    'not good',
    'poor work',
    'Could have done better.',
    'bad', 'weak', 'poor'
]
//const labels = [0.5, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

const labels = [0.5, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

function padEnd(array, minLength, fillValue = undefined) {
    return Object.assign(new Array(minLength).fill(fillValue), array);
}
function embed(d) {
    const v = padEnd(d.split(" ").map(v => {
        const i = vocabulary.indexOf(v.replace(/[^a-zA-Z ]/g, "").toLowerCase());
        return i >= 0 ? i : 0;
    }), maxLen, 0);
    console.log(v);
    return v;
}
const encodedDocs = docs.map(d => embed(d));
//console.log(encodedDocs);

const classify = (async (req, res) => {
    console.log("Load an existing model");
    let model = await load();
    if (!model) {
        console.log("model not found, train a new one");
        model = await train();
    }
    if (model) {
        console.log(req.body);
        const result = await getResult(req.body.text, model);
        console.log("predit result", result);
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
async function getResult(input, model) {
    let result = await model.predict(tf.tensor([embed(input)]));
    return await result.data();
}
async function load() {
    if (fs.existsSync(modelPath + '/model.json')) {
        return await tf.loadLayersModel('file://' + modelPath + '/model.json');
    }
    return null;
}
module.exports = {
    classify
}