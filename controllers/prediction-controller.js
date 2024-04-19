
const tf = require('@tensorflow/tfjs-node');
const fs = require("fs");
const modelPath = './prediction';
const { createModel, compileModel, fitModel, generatePath } = require('./model');

const map = [
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0, 1, 1],
    [0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
    [1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
];
const routes = [
    [[0, 0], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2], [8, 1], [9, 1]],
    [[0, 0], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [6, 2], [7, 2], [7, 3], [7, 4], [6, 4], [6, 5], [5, 5], [5, 6], [5, 7], [6, 7], [6, 8], [7, 8], [7, 9], [8, 9], [9, 9]],
]
let indices = [];
routes.forEach((r, i) => {
    r.forEach((p, i) => {
        const v = (p[0] * 10) + p[1];
        if (indices.indexOf(v) == -1) indices.push(v);
    });
});

function padEnd(array, minLength, fillValue = undefined) {
    return Object.assign(new Array(minLength).fill(fillValue), array);
}
const predict = (async (req, res) => {
    console.log("Load an existing model");
    let model = await load();
    if (!model) {
        console.log("model not found, train a new one");
        model = await train();
    }
    if (model) {
        let result = await generatePath(model, [0, 1, 11, 12], 20, 0.5);
        result = result.map(v=>[v/10,v%10]);
        res.send(result);
    }
})

async function train() {
    //Find all valid point index from route first

    const pointLen = indices.length;
    const rememberLen = 4;
    const model = createModel(rememberLen, pointLen, 16);
    compileModel(model, 1e-2);

    // Train the model.
    let input = new tf.TensorBuffer([
        routes.length, rememberLen, pointLen]);
    let label = new tf.TensorBuffer([routes.length, pointLen]);

console.log(indices);
    routes.forEach((r, i) => {
        let randomList = [];
        for (let i = 0;
            i < r.length - rememberLen - 1;
            i++) {
            randomList.push(i);
        }
        tf.util.shuffle(randomList);

        randomList.forEach((pi) => {
            let startIndex = pi;
            for (let j = 0; j < rememberLen; j++) {
                input.set(1, i, j, indices[startIndex + j]);
            }
            label.set(1, i, indices[startIndex + rememberLen]);
        });
    });

    await fitModel(
        model, input.toTensor(), label.toTensor(), 100, 50, 0.25,
        {
            onBatchEnd: async (batch, logs) => {
            },
            onEpochEnd: async (epoch, logs) => {
            }
        });

    //await model.save('file://' + modelPath);
    return model;
}
async function load() {
    if (fs.existsSync(modelPath + '/model.json')) {
        return await tf.loadLayersModel('file://' + modelPath + '/model.json');
    }
    return null;
}
module.exports = {
    predict
}