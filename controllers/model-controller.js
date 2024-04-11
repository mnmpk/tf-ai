
const tf = require('@tensorflow/tfjs-node');

const fs = require("fs");
const modelPath = '/Users/mma/git/tf-ai/model';
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
const encodedDocs = docs.map(d => tf.oneHot(docs.indexOf(d), docs.length).dataSync());

const index = (async (req, res) => {
    console.log("Load an existing model");
    let model = await load();
    if (!model) {
        console.log("model not found, train a new one");
        model = await train();
    }
    if (model) {
        const result = await getResult(model);
        //console.log("predit result", result);
        res.send(result);
    }
})
async function train() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

    // Create Training Data
    const xs = tf.tensor([0, 1, 2, 3, 4]);
    const ys = xs.mul(1.2).add(5);

    // Train the model.
    await model.fit(xs, ys, {
        epochs: 500,
        callbacks: {
            //onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
        }
    });
    await model.save('file://' + modelPath);
    return model;
}
async function load() {
    if (fs.existsSync(modelPath + '/model.json')) {
        return await tf.loadLayersModel('file://' + modelPath + '/model.json');
    }
    return null;
}
async function getResult(model) {
    const arr = [];
    for (let x = 0; x <= 10; x++) {
        arr.push({ x: x });
        let result = await model.predict(tf.tensor([Number(x)]));
        arr[x].y = Number(await result.data());
    }
    return arr;
}
function display(x, y) {
    console.log(x, y);
}
module.exports = {
    index
}