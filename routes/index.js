const express = require('express');

const router = express.Router();

const tf = require('@tensorflow/tfjs');

const {index} = require("../controllers/model-controller");
const {embedding} = require("../controllers/embedding-controller");

router.get('/', index);
router.get('/embedding', embedding);

module.exports = router;