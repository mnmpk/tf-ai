const tf = require('@tensorflow/tfjs-node');
const { Data, difficulty } = require('./data');


/**
 * Create a model for next-character prediction.
 * @param {number} sampleLen Sampling length: how many characters form the
 *   input to the model.
 * @param {number} charSetSize Size of the character size: how many unique
 *   characters there are.
 * @param {number|numbre[]} lstmLayerSizes Size(s) of the LSTM layers.
 * @return {tf.Model} A next-character prediction model with an input shape
 *   of `[null, sampleLen, charSetSize]` and an output shape of
 *   `[null, charSetSize]`.
 */
function createModel(sampleLen, charSetSize, lstmLayerSizes) {
  // Define input layers
  const timeSeriesInput = tf.input({ shape: [sampleLen, charSetSize] });
  const categoricalInput = tf.input({ shape: [difficulty.length] });

  // Feature extraction for time series data
  if (!Array.isArray(lstmLayerSizes)) {
    lstmLayerSizes = [lstmLayerSizes];
  }
  let last;
  for (let i = 0; i < lstmLayerSizes.length; ++i) {
    const lstmLayerSize = lstmLayerSizes[i];
    let lstm;
    if (last) {
      lstm = tf.layers.lstm({
        units: lstmLayerSize,
        returnSequences: i < lstmLayerSizes.length - 1,
        inputShape: i === 0 ? [sampleLen, charSetSize] : undefined
      }).apply(last);
    } else {
      lstm = tf.layers.lstm({
        units: lstmLayerSize,
        returnSequences: i < lstmLayerSizes.length - 1,
        inputShape: i === 0 ? [sampleLen, charSetSize] : undefined
      }).apply(timeSeriesInput);
    }
    last = tf.layers.dropout(0.5).apply(lstm);
  }
  const lstmDense = tf.layers.dense({ units: charSetSize, activation: 'softmax' }).apply(last);

  // Embedding for categorical data
  const embedding = tf.layers.embedding({ inputDim: difficulty.length, outputDim: 32 }).apply(categoricalInput);
  const flatten = tf.layers.flatten().apply(embedding);
  const categoricalDense = tf.layers.dense({ units: 1, activation: 'sigmoid' }).apply(flatten);

  // Concatenate the outputs
  const concat = tf.layers.concatenate().apply([lstmDense, categoricalDense]);

  // Additional hidden layers
  const hidden = tf.layers.dense({ units: 128, activation: 'relu' }).apply(concat);
  const output = tf.layers.dense({ units: charSetSize, activation: 'softmax' }).apply(hidden);

  // Create the model
  const model = tf.model({ inputs: [timeSeriesInput, categoricalInput], outputs: output });
  return model;
}

function compileModel(model, learningRate) {
  const optimizer = tf.train.rmsprop(learningRate);
  model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });
  console.log(`Compiled model with learning rate ${learningRate}`);
  model.summary();
}

/**
 * Train model.
 * @param {tf.Model} model The next-char prediction model, assumed to have an
 *   input shape of `[null, sampleLen, charSetSize]` and an output shape of
 *   `[null, charSetSize]`.
 * @param {TextData} textData The TextData object to use during training.
 * @param {number} numEpochs Number of training epochs.
 * @param {number} examplesPerEpoch Number of examples to draw from the
 *   `textData` object per epoch.
 * @param {number} batchSize Batch size for training.
 * @param {number} validationSplit Validation split for training.
 * @param {tf.CustomCallbackArgs} callbacks Custom callbacks to use during
 *   `model.fit()` calls.
 */
async function fitModel(
  model, data, numEpochs, batchSize, validationSplit,
  callbacks) {
  await model.fit(data.input, data.label, {
    epochs: numEpochs,
    batchSize: batchSize,
    validationSplit,
    callbacks
  });
}

const distance = (lastPoint, newPoint) => Math.hypot(newPoint[0] - lastPoint[0], newPoint[1] - lastPoint[1]);

async function generatePath(model, data, reqBody, temperature) {
  const {l, p, d} = reqBody;
  let path = p.map(p => parseInt(p));
  const rememberLen = model.inputs[0].shape[1];
  const indicesSize = model.inputs[0].shape[2];
  if (path.length > rememberLen) {
    path = path.slice(-rememberLen);
  } else if (path.length < rememberLen) {
    //Invalid input
    path = Object.assign(new Array(rememberLen).fill(path[path.length - 1]), path);
  } else {
    path = path.slice();
  }
  let generated = [];
  while (generated.length < l) {
    console.log(path);
    // Encode the current input sequence as a one-hot Tensor.
    const inputBuffer =
      new tf.TensorBuffer([1, rememberLen, indicesSize]);
    // Make the one-hot encoding of the seeding sentence.
    for (let i = 0; i < rememberLen; ++i) {
      console.log("start", path[i], data.encode(path[i]));
      inputBuffer.set(1, 0, i, data.encode(path[i]));
    }
    const input = inputBuffer.toTensor();

    await input.data().then(data => console.log("input", data));
    // Call model.predict() to get the probability values of the next
    // character.
    console.log("difficulty:"+d, difficulty.indexOf(d));
    let difficultyInput = [];
    switch (difficulty.indexOf(d)) {
      case 0: difficultyInput.push([1, 0, 0]); break;
      case 1: difficultyInput.push([0, 1, 0]); break;
      case 2: difficultyInput.push([0, 0, 1]); break;
  }
    const output = model.predict([input, tf.tensor2d(difficultyInput)]);

    await output.data().then(data => console.log("output", data));

    // Sample randomly based on the probability values.
    const winnerIndex = sample(tf.squeeze(output), temperature);

    const lastPoint = Data.conv2Coordinate(generated[generated.length - 1]);
    const newPoint = Data.conv2Coordinate(data.decode(winnerIndex));
    console.log("winner", winnerIndex, data.decode(winnerIndex), newPoint, "distance:" + distance(lastPoint, newPoint));
    if (distance(lastPoint, newPoint) > 1) {
      input.dispose();
      output.dispose();
      break;
    }
    generated.push(data.decode(winnerIndex));
    path = path.slice(1);
    path.push(data.decode(winnerIndex));

    // Memory cleanups.
    input.dispose();
    output.dispose();
  }
  console.log(generated);
  return generated;
}

/**
 * Generate text using a next-char-prediction model.
 *
 * @param {tf.Model} model The model object to be used for the text generation,
 *   assumed to have input shape `[null, sampleLen, charSetSize]` and output
 *   shape `[null, charSetSize]`.
 * @param {number[]} sentenceIndices The character indices in the seed sentence.
 * @param {number} length Length of the sentence to generate.
 * @param {number} temperature Temperature value. Must be a number >= 0 and
 *   <= 1.
 * @param {(char: string) => Promise<void>} onTextGenerationChar An optinoal
 *   callback to be invoked each time a character is generated.
 * @returns {string} The generated sentence.
 */
async function generateText(
  model, textData, sentenceIndices, length, temperature,
  onTextGenerationChar) {
  const sampleLen = model.inputs[0].shape[1];
  const charSetSize = model.inputs[0].shape[2];

  // Avoid overwriting the original input.
  sentenceIndices = sentenceIndices.slice();

  let generated = '';
  while (generated.length < length) {
    // Encode the current input sequence as a one-hot Tensor.
    const inputBuffer =
      new tf.TensorBuffer([1, sampleLen, charSetSize]);

    // Make the one-hot encoding of the seeding sentence.
    for (let i = 0; i < sampleLen; ++i) {
      inputBuffer.set(1, 0, i, sentenceIndices[i]);
    }
    const input = inputBuffer.toTensor();

    // Call model.predict() to get the probability values of the next
    // character.
    const output = model.predict(input);

    // Sample randomly based on the probability values.
    const winnerIndex = sample(tf.squeeze(output), temperature);
    const winnerChar = textData.getFromCharSet(winnerIndex);
    if (onTextGenerationChar != null) {
      await onTextGenerationChar(winnerChar);
    }

    generated += winnerChar;
    sentenceIndices = sentenceIndices.slice(1);
    sentenceIndices.push(winnerIndex);

    // Memory cleanups.
    input.dispose();
    output.dispose();
  }
  return generated;
}

/**
 * Draw a sample based on probabilities.
 *
 * @param {tf.Tensor} probs Predicted probability scores, as a 1D `tf.Tensor` of
 *   shape `[charSetSize]`.
 * @param {tf.Tensor} temperature Temperature (i.e., a measure of randomness
 *   or diversity) to use during sampling. Number be a number > 0, as a Scalar
 *   `tf.Tensor`.
 * @returns {number} The 0-based index for the randomly-drawn sample, in the
 *   range of `[0, charSetSize - 1]`.
 */
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