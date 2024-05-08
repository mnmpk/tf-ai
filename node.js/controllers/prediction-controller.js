
const tf = require('@tensorflow/tfjs-node');
const fs = require("fs");
const modelPath = './prediction';
const { createModel, compileModel, fitModel, generatePath } = require('./model');
const { trails, Data } = require('./data');

const rememberLen = 2;
const textMaxLength = 200;
const data = new Data(trails, rememberLen, textMaxLength);

const predict = (async (req, res) => {
    console.log("Load an existing model");
    let model = await load();
    if (!model) {
        console.log("model not found, train a new one");
        model = await train();
    }
    if (model) {
        let result = await generatePath(model, data, req.body, 0.5);
        result = result.map(v => Data.conv2Coordinate(v));
        res.send(result);
    }
})

async function train() {
    const model = createModel(rememberLen, data.pointLen, [64, 128], { maxLen: textMaxLength, embeddingSize: data.tags.length/*parseInt(data.model.size)*/ }, 2);
    compileModel(model, 1e-2);
    const d = await data.prepareData(1000);
    await fitModel(
        model, d, 100, 128, 0.0625,
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