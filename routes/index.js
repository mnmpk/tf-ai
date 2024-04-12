const express = require('express');

const router = express.Router();

const tf = require('@tensorflow/tfjs');

const {index} = require("../controllers/model-controller");
const {predict} = require("../controllers/embedding-controller");

router.get('/', index);
router.post('/predict', predict);

module.exports = router;