const express = require('express');

const router = express.Router();

const tf = require('@tensorflow/tfjs');

const {index} = require("../controllers/model-controller");
const {classify} = require("../controllers/embedding-controller");
const {predict} = require("../controllers/prediction-controller");

router.get('/', index);
router.post('/classify', classify);
router.post('/predict', predict);

module.exports = router;