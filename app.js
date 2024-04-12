const express = require('express');
const routes = require('./routes/index');

const app = express();
app.use(express.json());
app.use(express.urlencoded());
app.use('/', routes);
app.use(express.static('public'));

module.exports = app;