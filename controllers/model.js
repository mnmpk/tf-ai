const tf = require('@tensorflow/tfjs-node');


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
  if (!Array.isArray(lstmLayerSizes)) {
    lstmLayerSizes = [lstmLayerSizes];
  }

  const model = tf.sequential();
  for (let i = 0; i < lstmLayerSizes.length; ++i) {
    const lstmLayerSize = lstmLayerSizes[i];
    model.add(tf.layers.lstm({
      units: lstmLayerSize,
      returnSequences: i < lstmLayerSizes.length - 1,
      inputShape: i === 0 ? [sampleLen, charSetSize] : undefined
    }));
  }
  model.add(
      tf.layers.dense({units: charSetSize, activation: 'softmax'}));

  return model;
}

function compileModel(model, learningRate) {
  const optimizer = tf.train.rmsprop(learningRate);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
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
    model, input, label, numEpochs, batchSize, validationSplit,
    callbacks) {
  for (let i = 0; i < numEpochs; ++i) {
    await model.fit(input, label, {
      epochs: 1,
      batchSize: batchSize,
      validationSplit,
      callbacks
    });
  }
}
async function generatePath(model, path, length,temperature){

  const rememberLen = model.inputs[0].shape[1];
  const indicesSize = model.inputs[0].shape[2];
  
  path = path.slice();
  let generated = [];
  while (generated.length < length) {
    console.log(path);
    // Encode the current input sequence as a one-hot Tensor.
    const inputBuffer =
        new tf.TensorBuffer([1, rememberLen, indicesSize]);
    // Make the one-hot encoding of the seeding sentence.
    for (let i = 0; i < rememberLen; ++i) {
      console.log("start",path[i]);
      inputBuffer.set(1, 0, i, path[i]);
    }
    const input = inputBuffer.toTensor();

    // Call model.predict() to get the probability values of the next
    // character.
    const output = model.predict(input);
    // Sample randomly based on the probability values.
    const winnerIndex = sample(tf.squeeze(output), temperature);

    console.log(winnerIndex);
    generated.push(winnerIndex);
    path = path.slice(1);
    path.push(winnerIndex);

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