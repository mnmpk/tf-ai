const express = require('express');

const router = express.Router();

const tf = require('@tensorflow/tfjs');

const {index} = require("../controllers/model-controller");
const {classify} = require("../controllers/embedding-controller");
const {predict} = require("../controllers/prediction-controller");
const {s2sPredict} = require("../controllers/s2s-controller");
const {call} = require("../controllers/transformer-controller");


router.get('/', index);
router.post('/classify', classify);
router.post('/predict', predict);
router.post('/s2s', s2sPredict);
router.post('/transformer', call);

module.exports = router;