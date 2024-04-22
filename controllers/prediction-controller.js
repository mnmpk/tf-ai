
const tf = require('@tensorflow/tfjs-node');
const fs = require("fs");
const modelPath = './prediction';
const { createModel, compileModel, fitModel, generatePath } = require('./model');
const { trails, Data, tags, animals, facilities, difficulty, landscape } = require('./data');

const rememberLen = 2;
const data = new Data(trails, rememberLen);

const predict = (async (req, res) => {
    console.log("Load an existing model");
    let model = await load();
    if (!model) {
        console.log("model not found, train a new one");
        model = await train();
    }
    if (model) {
        let result = await generatePath(model, data, req.body, 1.0);
        result = result.map(v => Data.conv2Coordinate(v));
        res.send(result);
    }
})

async function train() {
    const model = createModel(rememberLen, data.pointLen, [64,128], [tags.length, animals.length, facilities.length], 2);
    compileModel(model, 1e-2);
    const d = await data.prepareData(100);
    await fitModel(
        model, d, 300, 128, 0.0625,
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