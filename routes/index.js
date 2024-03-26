const express = require('express');

const router = express.Router();

const tf = require('@tensorflow/tfjs');
router.get('/', (req, res) => {
  // Create Training Data
  const xs = tf.tensor([0, 1, 2, 3, 4]);
  const ys = xs.mul(1.2).add(5);

  // Define a Linear Regression Model
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  // Specify Loss and Optimizer
  model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

  // Train the Model
  model.fit(xs, ys, { epochs: 500 }).then(() => { myFunction() });

  // Use the Model
  function myFunction() {
    const arr = [];
    for (let x = 0; x <= 10; x++) {
      arr.push({x:x});
      let result = model.predict(tf.tensor([Number(x)]));
      result.data().then(y => {
        arr[x].y=Number(y);
        if (x == 10) { res.send(arr); };
      });
    }
  }
  
});

module.exports = router;