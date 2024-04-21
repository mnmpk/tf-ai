
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
    [[0, 0], [1, 0], [1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [6, 2], [7, 2], [7, 3], [7, 4], [6, 4], [6, 5], [5, 5], [5, 6], [5, 7], [6, 7], [6, 8], [7, 8], [7, 9], [8, 9], [9, 9]],
]

let indices = [];
routes.forEach((r, i) => {
    r.forEach((p, i) => {
        const v = (p[1] * 10) + p[0];
        if (indices.indexOf(v) == -1) indices.push(v);
    });
});
console.log("indices", indices);

const predict = (async (req, res) => {
    console.log("Load an existing model");
    let model = await load();
    if (!model) {
        console.log("model not found, train a new one");
        model = await train();
    }
    if (model) {
        let result = await generatePath(model, indices, req.body.path.map(p => parseInt(p)), req.body.length, 0.8);
        result = result.map(v => [v % 10, Math.floor(v / 10)]);
        res.send(result);
    }
})

async function train() {
    //Find all valid point index from route first
    const routeExampleSize = 300;
    const pointLen = indices.length;
    const rememberLen = 3;
    const model = createModel(rememberLen, pointLen, [64,128]);
    compileModel(model, 1e-2);

    // Train the model.
    let input = new tf.TensorBuffer([
        routes.length * routeExampleSize, rememberLen, pointLen]);
    let label = new tf.TensorBuffer([routes.length * routeExampleSize, pointLen]);

    routes.forEach((r, ri) => {
        let randomList = [];
        for (let i = 0;
            i < r.length - rememberLen;
            i++) {
            randomList.push(i);
        }
        tf.util.shuffle(randomList);
        console.log(randomList);

        for (let i = ri * routeExampleSize; i < ri * routeExampleSize + routeExampleSize; i++) {
            const routeStartIndex = randomList[(i%routeExampleSize) % randomList.length];
            for (let j = 0; j < rememberLen; j++) {
                const routePointIndex = routeStartIndex + j;
                const worldValue = (r[routePointIndex][1]*10)+r[routePointIndex][0];
                console.log("i:"+i, "j:"+j, "route point index:"+routePointIndex, "route point:"+r[routePointIndex], "world value:"+worldValue, "world index:"+indices.indexOf(worldValue));
                input.set(1, i, j, indices.indexOf(worldValue));
            }
            const targetPointIndex = routeStartIndex + rememberLen;
            const targetWorldValue = (r[targetPointIndex][1]*10)+r[targetPointIndex][0];
            console.log("i:"+i, "target point index:"+targetPointIndex, "target point:"+r[targetPointIndex], "world value:"+targetWorldValue, "world index:"+indices.indexOf(targetWorldValue));
            label.set(1, i, indices.indexOf(targetWorldValue));
        }
    });
    //input.toTensor().data().then(data => console.log(data));
    //label.toTensor().data().then(data => console.log(data));
    //console.log(input, label);

    await fitModel(
        model, input.toTensor(), label.toTensor(), 300, 128, 0.0625,
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