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
class Data {

    constructor(routes, rememberLen) {
        this.routes = routes;
        this.rememberLen = rememberLen;
        this.routesIndices = [];
        this.routes.forEach((r, i) => {
            r.forEach((p, i) => {
                const v = (p[1] * 10) + p[0];
                if (this.routesIndices.indexOf(v) == -1) this.routesIndices.push(v);
            });
        });
        this.pointLen = this.routesIndices.length;
    }

    static conv2Coordinate(v) {
        return [v % 10, Math.floor(v / 10)];
    }
    static conv2Value(c) {
        return (c[1] * 10) + c[0];
    }
    encode(v) {
        return this.routesIndices.indexOf(v);
    }
    decode(i) {
        return this.routesIndices[i];
    }
    async prepareData(examplePerRoute) {
        let input = new tf.TensorBuffer([
            this.routes.length * examplePerRoute, this.rememberLen, this.pointLen]);
        let label = new tf.TensorBuffer([this.routes.length * examplePerRoute, this.pointLen]);
        this.routes.forEach((r, ri) => {
            let randomList = [];
            for (let i = 0;
                i < r.length - this.rememberLen;
                i++) {
                randomList.push(i);
            }
            tf.util.shuffle(randomList);
            console.log(randomList);

            for (let i = ri * examplePerRoute; i < ri * examplePerRoute + examplePerRoute; i++) {
                const routeStartIndex = randomList[(i % examplePerRoute) % randomList.length];
                for (let j = 0; j < this.rememberLen; j++) {
                    const routePointIndex = routeStartIndex + j;
                    console.log("i:" + i, "j:" + j, "route point index:" + routePointIndex, "route point:" + r[routePointIndex]);
                    input.set(1, i, j, this.encode(Data.conv2Value(r[routePointIndex])));
                }
                const targetPointIndex = routeStartIndex + this.rememberLen;
                console.log("i:" + i, "target point index:" + targetPointIndex, "target point:" + r[targetPointIndex]);
                label.set(1, i, this.encode(Data.conv2Value(r[targetPointIndex])));
            }
        });
        return { input: input.toTensor(), label: label.toTensor() };
    }
}



module.exports = { map, routes, Data }