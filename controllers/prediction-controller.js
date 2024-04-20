
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
    [[0, 0], [1, 0], [1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2], [8, 1], [9, 1]],
    [[0, 0], [1, 0], [1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [6, 2], [7, 2]],
    [[2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2], [8, 1], [9, 1]],
    [[2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [6, 2], [7, 2]],
    [[1, 0], [1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [6, 2], [7, 2]],
    [[6, 2], [7, 2], [8, 2], [8, 1], [9, 1]],
    [[1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [6, 2], [7, 2]],
    [[0, 0], [1, 0], [1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2], [8, 1], [9, 1]],
    [[2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2], [8, 1], [9, 1]],
    [[0, 0], [1, 0], [1, 1], [2, 1], [3, 1]],
    //[[0, 0], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [6, 2], [7, 2], [7, 3], [7, 4], [6, 4], [6, 5], [5, 5], [5, 6], [5, 7], [6, 7], [6, 8], [7, 8], [7, 9], [8, 9], [9, 9]],
]

const r = routes[0];
let indices = [];
//routes.forEach((r, i) => {
r.forEach((p, i) => {
    const v = (p[1] * 10) + p[0];
    if (indices.indexOf(v) == -1) indices.push(v);
});
//});
console.log(indices);

const predict = (async (req, res) => {
    console.log("Load an existing model");
    let model = await load();
    if (!model) {
        console.log("model not found, train a new one");
        model = await train();
    }
    if (model) {
        let result = await generatePath(model, indices, req.body.path.map(p => parseInt(p)), req.body.length, 0.1);
        result = result.map(v => [v % 10, Math.floor(v / 10)]);
        res.send(result);
    }
})

async function train() {
    //routes.forEach((r, i) => {
    //});
    //Find all valid point index from route first
    const routeExampleSize = 100;
    const pointLen = indices.length;
    const rememberLen = 2;
    const model = createModel(rememberLen, pointLen, 32);
    compileModel(model, 1e-2);

    // Train the model.
    let input = new tf.TensorBuffer([
        routes.length, rememberLen, pointLen]);
    let label = new tf.TensorBuffer([routes.length, pointLen]);

    let randomList = [];
    for (let i = 0;
        i < r.length - rememberLen;
        i++) {
        randomList.push(i);
    }
    tf.util.shuffle(randomList);
    console.log(randomList);

    for (let i = 0; i < routeExampleSize; i++) {
        const startIndex = randomList[i % randomList.length];
        for (let j = 0; j < rememberLen; j++) {
            const p = startIndex + j;
            console.log(i, j, startIndex + j, p, [p % 10, Math.floor(p / 10)]);
            input.set(1, i, j, startIndex + j);
        }
        const t = startIndex + rememberLen;
        console.log("target", startIndex + rememberLen, t, [t % 10, Math.floor(t / 10)]);
        label.set(1, i, startIndex + rememberLen);
    }
    //input.toTensor().data().then(data => console.log(data));
    //label.toTensor().data().then(data => console.log(data));
    //console.log(input, label);

    /*console.log("indices", indices);
    routes.forEach((r, i) => {
        let randomList = [];
        for (let i = 0;
            i < r.length - rememberLen - 1;
            i++) {
            randomList.push(i);
        }
        tf.util.shuffle(randomList);
        console.log("randomList", randomList);
        randomList.forEach((pi) => {
            let startIndex = pi;
            for (let j = 0; j < rememberLen; j++) {
                const p = indices[startIndex + j];
                console.log(i, j, startIndex + j, p, [Math.floor(p / 10), p % 10]);
                input.set(1, i, j, indices[startIndex + j]);
            }
            const t = indices[startIndex + rememberLen];
            console.log("target", startIndex + rememberLen, t, [Math.floor(t / 10), t % 10]);
            label.set(1, i, indices[startIndex + rememberLen]);
        });
        console.log(input, label);
    });*/

    await fitModel(
        model, input.toTensor(), label.toTensor(), 1000, 128, 0.0625,
        {
            onBatchEnd: async (batch, logs) => {
            },
            onEpochEnd: async (epoch, logs) => {
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
module.exports = {
    predict
}