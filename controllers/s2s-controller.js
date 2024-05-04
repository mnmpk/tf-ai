
const tf = require('@tensorflow/tfjs-node');
const fs = require("fs");
const modelPath = './s2s/model';
const { createModel, compileModel, fitModel, generatePath } = require('../s2s/model');
const { trails, Data } = require('../s2s/data');

const rememberSize = 2;
const textMaxLength = 10;
const data = new Data(trails, textMaxLength, rememberSize);

const s2sPredict = (async (req, res) => {
    console.log("Load an existing model");
    let model = await load();
    if (!model) {
        console.log("model not found, train a new one");
        model = await train();
    }
    if (model) {
        let result = await generatePath(model, data, req.body);

        result = result.map(v => Data.conv2Coordinate(v));
        res.send(result);
    }
})

async function train() {
    const model = createModel(textMaxLength, data.vocab.length/*, parseInt(data.w2vModel.size)*/, 32, data.pointSize);
    compileModel(model, 1e-2);
    const d = await data.prepareData(1000);
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
    s2sPredict
}