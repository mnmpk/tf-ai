
const tf = require('@tensorflow/tfjs-node');
const fs = require("fs");
const modelPath = './prediction';
const { createModel, compileModel, fitModel } = require('./model');

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
        let result = await model.predict();
        res.send(result);
    }
})

async function train() {
    const pointLen = 1;
    const sampleLen = 10;
    const model = createModel(sampleLen, pointLen, 16);
    compileModel(model, 1e-2);

    // Train the model.
    let input = [];
    let label = []
    routes.forEach((r, i) => {
        let temp = [];
        let tempL = [];
        for(j=0;j<sampleLen;j++){
            const startIndex = Math.floor(Math.random()*r.length)-1;
            const p = r[startIndex<0?0:startIndex];
            const n = r[(startIndex<0?0:startIndex)+1];
            console.log(startIndex, p, n);
            temp.push([(p[0] * 10) + p[1]]);
            tempL.push((n[0] * 10) + n[1]);
        }
        input.push(temp);
        label.push(tempL);

        /*input.push([
            padEnd(r.slice(0, r.length - 1).map(p => {
                return (p[0] * 10) + p[1];
            }), pointLen, -1)
        ]);
        label.push(
            padEnd(r.slice(1, r.length).map(p => {
                return (p[0] * 10) + p[1];
            }), pointLen, -1)
        );*/
        /*const tmp = r.map(p => {
            p.unshift((p[0] * 10) + p[1]);
            return p;
        });
        input.push(tmp.slice(0, tmp.length - 1));
        label.push(tmp.slice(1, tmp.length));*/
    });
    console.log(input);
    console.log(label);

    await fitModel(
        model, input, label, 2, 16, 4, 0.25,
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