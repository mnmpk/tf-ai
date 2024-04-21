
const tf = require('@tensorflow/tfjs-node');
const fs = require("fs");
const modelPath = './prediction';
const { createModel, compileModel, fitModel, generatePath } = require('./model');
const { routesIndices, prepareData } = require('./data');

const predict = (async (req, res) => {
    console.log("Load an existing model");
    let model = await load();
    if (!model) {
        console.log("model not found, train a new one");
        model = await train();
    }
    if (model) {
        let result = await generatePath(model, routesIndices, req.body.path.map(p => parseInt(p)), req.body.length, 1.0);
        result = result.map(v => [v % 10, Math.floor(v / 10)]);
        res.send(result);
    }
})

async function train() {
    //Find all valid point index from route first
    const pointLen = routesIndices.length;
    const rememberLen = 3;
    const model = createModel(rememberLen, pointLen, [64,128]);
    compileModel(model, 1e-2);

    await fitModel(
        model, await prepareData(rememberLen, pointLen), 300, 128, 0.0625,
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