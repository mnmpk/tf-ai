const tf = require('@tensorflow/tfjs-node');

// Define the Transformer model
class Transformer {
  constructor(vocabSize, numClasses, numLayers, numHeads, hiddenDim, dropoutRate) {
    this.vocabSize = vocabSize;
    this.numClasses = numClasses;
    this.numLayers = numLayers;
    this.numHeads = numHeads;
    this.hiddenDim = hiddenDim;
    this.dropoutRate = dropoutRate;
    this.model = this.build();
  }

  build() {
    const input = tf.input({ shape: [null] });
    const embedding = tf.layers.embedding({ inputDim: this.vocabSize, outputDim: this.hiddenDim }).apply(input);
    const positionalEncoding = this.getPositionalEncoding(embedding.shape[2]);
    const x = tf.layers.concatenate().apply([embedding, positionalEncoding]);
    const dropout = tf.layers.dropout({ rate: this.dropoutRate }).apply(x);

    let attentionOutput = dropout;
    for (let i = 0; i < this.numLayers; i++) {
      attentionOutput = this.multiHeadAttention(attentionOutput);
      attentionOutput = tf.layers.add([attentionOutput, dropout]);
      attentionOutput = tf.layers.normalization({ axis: -1 }).apply(attentionOutput);

      const feedForwardOutput = this.feedForward(attentionOutput);
      attentionOutput = tf.layers.add([attentionOutput, feedForwardOutput]);
      attentionOutput = tf.layers.normalization({ axis: -1 }).apply(attentionOutput);
    }

    const flatten = tf.layers.flatten(attentionOutput);
    const dense = tf.layers.dense({ units: this.numClasses, activation: 'softmax' }).apply(flatten);

    return tf.model({ inputs: input, outputs: dense });
  }

  getPositionalEncoding(sequenceLength) {
    /*const position = tf.linspace(0, sequenceLength - 1, sequenceLength);
    const divTerm = tf.pow(10000, tf.linspace(0, this.hiddenDim - 1, this.hiddenDim).div(this.hiddenDim));
    const divTermExpanded = divTerm.expandDims(0);

    const angle = position.expandDims(1).div(divTermExpanded);
    const sinValues = tf.sin(angle.slice([0, 0, tf.newaxis], [sequenceLength, this.hiddenDim, 1]));
    const cosValues = tf.cos(angle.slice([0, 0, tf.newaxis], [sequenceLength, this.hiddenDim, 1]));

    return tf.layers.concatenate().apply([sinValues, cosValues], { axis: -1 });*/

    const position = tf.linspace(0, sequenceLength - 1, sequenceLength);
    const divTerm = tf.pow(10000, tf.linspace(0, this.hiddenDim - 1, this.hiddenDim).div(this.hiddenDim));
    const divTermExpanded = divTerm.expandDims(0);

    const angle = position.expandDims(1).div(divTermExpanded);
    const sinValues = tf.sin(angle.slice([0, 0], [sequenceLength, 1]));
    const cosValues = tf.cos(angle.slice([0, 0], [sequenceLength, 1]));

    return tf.layers.concatenate().apply([sinValues, cosValues]);
  }

  multiHeadAttention(inputs) {
    const query = tf.layers.dense({ units: this.hiddenDim })(inputs);
    const key = tf.layers.dense({ units: this.hiddenDim })(inputs);
    const value = tf.layers.dense({ units: this.hiddenDim })(inputs);

    const scaledDotProduct = tf.layers.dot({ axes: [-1, -1] }).apply([query, key]);
    const attentionWeights = tf.layers.activation({ activation: 'softmax' }).apply(scaledDotProduct);
    const attentionOutput = tf.layers.dot({ axes: [2, 1] }).apply([attentionWeights, value]);

    return attentionOutput;
  }

  feedForward(inputs) {
    const hidden = tf.layers.dense({ units: this.hiddenDim * 4, activation: 'relu' })(inputs);
    const output = tf.layers.dense({ units: this.hiddenDim })(hidden);

    return output;
  }

  summary() {
    this.model.summary();
  }
}

module.exports = {
  Transformer
}