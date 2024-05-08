const tf = require('@tensorflow/tfjs-node');
const { Data } = require('./data');

function createModel(vocabSize, latentDim, pointSize) {
  const embeddingInput = tf.input({
    shape: [null, vocabSize],
    name: 'embeddingInput',
  });
  //const embedding = tf.layers.embedding({ inputDim: vocabSize, outputDim: 32 }).apply(embeddingInput);
  //const flatten = tf.layers.flatten().apply(embedding);
  //const dense = tf.layers.dense({ units: 1, activation: 'sigmoid' }).apply(embedding);
  const encoder = tf.layers.lstm({
    units: latentDim,
    returnState: true,
    name: 'encoderLstm',
  });
  const [, stateH, stateC] = encoder.apply(embeddingInput);
  const encoderStates = [stateH, stateC];

  const decoderInputs = tf.layers.input({
    shape: [null, pointSize],
    name: 'decoderInputs',
  });
  const decoderLstm = tf.layers.lstm({
    units: latentDim,
    returnSequences: true,
    returnState: true,
    name: 'decoderLstm',
  });
  const [decoderOutputs,] = decoderLstm.apply(
    [decoderInputs, ...encoderStates],
  );
  const decoderDense = tf.layers.dense({
    units: pointSize,
    activation: 'softmax',
    name: 'decoderDense',
  });
  const decoderDenseOutputs = decoderDense.apply(decoderOutputs);

  const model = tf.model({
    inputs: [embeddingInput, decoderInputs],
    outputs: decoderDenseOutputs,
    name: 'seq2seqModel',
  });
  return model;
}

function compileModel(model, learningRate) {
  const optimizer = tf.train.rmsprop(learningRate);
  model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });
  console.log(`Compiled model with learning rate ${learningRate}`);
  model.summary();
}

async function fitModel(
  model, data, numEpochs, batchSize, validationSplit,
  callbacks) {
  await model.fit(data.input, data.target, {
    epochs: numEpochs,
    batchSize: batchSize,
    validationSplit,
    callbacks
    /*: args.logDir == null ? null :
          tfn.node.tensorBoard(args.logDir, {
            updateFreq: args.logUpdateFreq
          })*/
  });
}

async function generatePath(model, data, reqBody) {
  const maxLength = model.inputs[1].shape[1];
  const indicesSize = model.inputs[1].shape[2];

  //this.numEncoderTokens = model.input[0].shape[2];
  //this.numDecoderTokens = model.input[1].shape[2];

  const encoderModel = prepareEncoderModel(model);
  const decoderModel = prepareDecoderModel(model);

  const { l, p, d, v, desc } = reqBody;
  //const Segmenter = require('node-analyzer');
  //const segmenter = new Segmenter();
  //let arr = new Array(data.textMaxSize).fill(new Array(parseInt(data.w2vModel.size)).fill(0));
  //arr = Object.assign(arr, data.w2vModel.getVectors(segmenter.analyze(desc) || new Array(parseInt(data.w2vModel.size)).fill(0)).map(v => v.values));
  //const words = segmenter.analyze(desc).split(" ");
  //const words = desc.split(" ");

  let words = [];
  data.vocab.forEach(v => {
    if (desc.indexOf(v) >= 0)
      words.push(v);
  });

  let encorderInput = new tf.TensorBuffer([1, data.textMaxSize, data.vocab.length]);
  words.forEach((w, i) => {
    const index = data.vocab.indexOf(w.toLowerCase());
    console.log(w, index);
    encorderInput.set(1, 0, i, index >= 0 ? index : 0);
  });
  //console.log("encorderInput", encorderInput.toTensor().dataSync());
  // Encode the inputs state vectors.
  let statesValue = encoderModel.predict(encorderInput.toTensor());
  // Generate empty target sequence of length 1.
  let targetSeq = tf.buffer([1, p.length, indicesSize]);
  // Populate the first character of the target sequence with the start
  // character.
  let lastIndex = 0;
  p.forEach((point, i) => {
    lastIndex = data.encode(parseInt(point));
    targetSeq.set(1, 0, i, lastIndex);
    console.log("input", lastIndex, Data.conv2Coordinate(parseInt(point)));
  });
  //console.log("input", targetSeq.toTensor().arraySync());

  // Sample loop for a batch of sequences.
  // (to simplify, here we assume that a batch of size 1).
  let stopCondition = false;
  let decodedSentence = [];
  while (!stopCondition) {
    const predictOutputs =
      decoderModel.predict([targetSeq.toTensor()].concat(statesValue));
    const outputTokens = predictOutputs[0];
    const h = predictOutputs[1];
    const c = predictOutputs[2];

    //console.log("output", outputTokens.arraySync());
    // Sample a token.
    // We know that outputTokens.shape is [1, 1, n], so no need for slicing.
    const logits = outputTokens.reshape([outputTokens.shape[1], outputTokens.shape[2]]);
    const results = logits.unstack();
    const sampledTokenIndex = results[results.length-1].argMax().dataSync()[0];
    //const logits = outputTokens.reshape([outputTokens.shape[1]]);
    //const sampledTokenIndex = logits.argMax().dataSync()[0];
    const sampledChar = data.decode(sampledTokenIndex);
    decodedSentence.push(sampledChar);
    console.log("output", sampledChar, Data.conv2Coordinate(sampledChar));
    if (sampledChar === -1 ||
      decodedSentence.length > data.maxLength ||
      decodedSentence.length > l) {
      stopCondition = true;
    }
    newTargetSeq = tf.buffer([1, p.length, indicesSize]);
    for (var i = 1; i < p.length; i++) {
      let index = 0;
      for (var j = 0; j < indicesSize; j++) {
        const temp = targetSeq.get(0, i, j);
        if (temp == 1)
          index = j;
      }
      newTargetSeq.set(1, 0, i - 1, index);
    }
    newTargetSeq.set(1, 0, p.length - 1, sampledTokenIndex);
    targetSeq = newTargetSeq;
    lastIndex = sampledTokenIndex;
    /*const logits = outputTokens.reshape([outputTokens.shape[1], outputTokens.shape[2]]);
    const results = logits.unstack();
    // Update the target sequence (of length 1).
    targetSeq = tf.buffer([1, p.length, indicesSize]);
    results.forEach((r, i) => {
      const sampledTokenIndex = r.argMax().dataSync()[0];
      //const sampledTokenIndex = sample(tf.squeeze(outputTokens), 0.3);
      const sampledChar = data.decode(sampledTokenIndex);
      decodedSentence.push(sampledChar);
      console.log("output", Data.conv2Coordinate(sampledChar));
      // Exit condition: either hit max length or find stop character.
      if (sampledChar === -1 ||
        decodedSentence.length > data.maxLength ||
        decodedSentence.length > l) {
        stopCondition = true;
      }
      targetSeq.set(1, 0, i, sampledTokenIndex);
      lastIndex = sampledTokenIndex;
      // Update states.
      //statesValue = [h, c];
    });*/


    // Update states.
    //statesValue = [h, c];
  }
  return decodedSentence;
  /*
  let path = p.map(p => parseInt(p));
  const maxLength = model.inputs[1].shape[1];
  const indicesSize = model.inputs[1].shape[2];
  if (path.length > maxLength) {
    path = path.slice(-maxLength);
  } else if (path.length < maxLength) {
    //Invalid input
    path = Object.assign(new Array(maxLength).fill(path[path.length - 1]), path);
  } else {
    path = path.slice();
  }
  let generated = [];
  while (generated.length < l) {
    console.log(path);
    // Encode the current input sequence as a one-hot Tensor.
    const inputBuffer =
      new tf.TensorBuffer([1, maxLength, indicesSize]);
    // Make the one-hot encoding of the seeding sentence.
    for (let i = 0; i < maxLength; ++i) {
      console.log("start", path[i], data.encode(path[i]));
      inputBuffer.set(1, 0, i, data.encode(path[i]));
    }
    const input = inputBuffer.toTensor();
    const Segmenter = require('node-analyzer');
    const segmenter = new Segmenter();
    let arr = new Array(data.textMaxSize).fill(new Array(parseInt(data.w2vModel.size)).fill(0));
    arr = Object.assign(arr, data.w2vModel.getVectors(segmenter.analyze(desc) || new Array(parseInt(data.w2vModel.size)).fill(0)).map(v => v.values));
    const output = model.predict([tf.tensor3d([arr]), input]);
    //await output.data().then(data => console.log("output", data));

    const o = output.arraySync()[0][0];//.forEach(o => {
      const winnerIndex = sample(tf.squeeze(o), temperature);
      const lastPoint = Data.conv2Coordinate(generated[generated.length - 1]);
      const newPoint = Data.conv2Coordinate(data.decode(winnerIndex));
      console.log("winner", winnerIndex, data.decode(winnerIndex), newPoint, "distance:" + distance(lastPoint, newPoint));
      generated.push(data.decode(winnerIndex));
      path = path.slice(1);
      path.push(data.decode(winnerIndex));
    //});

    // Memory cleanups.
    input.dispose();
    output.dispose();
  }
  return generated;*/
}

function prepareEncoderModel(model) {

  const encoderInputs = model.input[0];
  const stateH = model.layers[2].output[1];
  const stateC = model.layers[2].output[2];
  const encoderStates = [stateH, stateC];

  return tf.model({ inputs: encoderInputs, outputs: encoderStates });
}

function prepareDecoderModel(model) {

  const stateH = model.layers[2].output[1];
  const latentDim = stateH.shape[stateH.shape.length - 1];
  console.log('latentDim = ' + latentDim);
  const decoderStateInputH =
    tf.input({ shape: [latentDim], name: 'decoder_state_input_h' });
  const decoderStateInputC =
    tf.input({ shape: [latentDim], name: 'decoder_state_input_c' });
  const decoderStateInputs = [decoderStateInputH, decoderStateInputC];

  const decoderLSTM = model.layers[3];
  const decoderInputs = decoderLSTM.input[0];
  const applyOutputs =
    decoderLSTM.apply(decoderInputs, { initialState: decoderStateInputs });
  let decoderOutputs = applyOutputs[0];
  const decoderStateH = applyOutputs[1];
  const decoderStateC = applyOutputs[2];
  const decoderStates = [decoderStateH, decoderStateC];

  const decoderDense = model.layers[4];
  decoderOutputs = decoderDense.apply(decoderOutputs);
  return tf.model({
    inputs: [decoderInputs].concat(decoderStateInputs),
    outputs: [decoderOutputs].concat(decoderStates)
  });
}

const distance = (lastPoint, newPoint) => Math.hypot(newPoint[0] - lastPoint[0], newPoint[1] - lastPoint[1]);

function sample(probs, temperature) {
  return tf.tidy(() => {
    const logits = tf.div(tf.log(probs), Math.max(temperature, 1e-6));
    const isNormalized = false;
    // `logits` is for a multinomial distribution, scaled by the temperature.
    // We randomly draw a sample from the distribution.
    return tf.multinomial(logits, 1, null, isNormalized).dataSync()[0];
  });
}

module.exports = {
  createModel, compileModel, fitModel, generatePath
}