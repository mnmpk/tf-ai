const tf = require('@tensorflow/tfjs-node');
const map = [
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0, 1, 1],
    [0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
    [1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
];

const routes = [
    [[0, 0], [1, 0], [1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2], [8, 1], [9, 1]],
    [[0, 0], [1, 0], [1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [6, 2], [7, 2], [7, 3], [7, 4], [6, 4], [6, 5], [5, 5], [5, 6], [5, 7], [6, 7], [6, 8], [7, 8], [7, 9], [8, 9], [9, 9]],
    [[0, 0], [1, 0], [1, 1], [1, 2], [1, 3], [0, 3], [0, 4], [0, 5], [1, 5], [2, 5], [3, 5], [3, 6], [3, 7], [2, 7], [2, 8], [2, 9]],
    [[0, 0], [1, 0], [1, 1], [1, 2], [1, 3], [0, 3], [0, 4], [0, 5], [1, 5], [2, 5], [3, 5], [4, 5], [5, 5], [5, 6], [5, 7], [6, 7], [6, 8], [7, 8], [7, 9], [8, 9], [9, 9]],
]

const routesIndices = [];
routes.forEach((r, i) => {
    r.forEach((p, i) => {
        const v = (p[1] * 10) + p[0];
        if (routesIndices.indexOf(v) == -1) routesIndices.push(v);
    });
});

const routeExampleSize = 300;

function conv2Coordinate(v) {
    return [v % 10, Math.floor(v / 10)];
}
function conv2Value(c) {
    return (c[1] * 10) + c[0];
}
function encode(v) {
    return routesIndices.indexOf(v);
}
function decode(i) {
    return routesIndices[i];
}

async function prepareData(rememberLen, pointLen) {
    let input = new tf.TensorBuffer([
        routes.length * routeExampleSize, rememberLen, pointLen]);
    let label = new tf.TensorBuffer([routes.length * routeExampleSize, pointLen]);
    routes.forEach((r, ri) => {
        let randomList = [];
        for (let i = 0;
            i < r.length - rememberLen;
            i++) {
            randomList.push(i);
        }
        tf.util.shuffle(randomList);
        console.log(randomList);

        for (let i = ri * routeExampleSize; i < ri * routeExampleSize + routeExampleSize; i++) {
            const routeStartIndex = randomList[(i % routeExampleSize) % randomList.length];
            for (let j = 0; j < rememberLen; j++) {
                const routePointIndex = routeStartIndex + j;
                console.log("i:" + i, "j:" + j, "route point index:" + routePointIndex, "route point:" + r[routePointIndex]);
                input.set(1, i, j, encode(conv2Value(r[routePointIndex])));
            }
            const targetPointIndex = routeStartIndex + rememberLen;
            console.log("i:" + i, "target point index:" + targetPointIndex, "target point:" + r[targetPointIndex]);
            label.set(1, i, encode(conv2Value(r[targetPointIndex])));
        }
    });
    return { input: input.toTensor(), label: label.toTensor() };
}

module.exports = {
    map,
    routes,
    routesIndices,
    prepareData
}