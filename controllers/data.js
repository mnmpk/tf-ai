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

const trails = [
    {
        difficulty: 1,
        landscape: 3,
        description: "this is a beautiful trial with sea view, very easy to access.",
        route: [[0, 0], [1, 0], [1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2], [8, 1], [9, 1]],
    },
    {
        difficulty: 2,
        landscape: 5,
        description: "Very long trail!",
        route: [[0, 0], [1, 0], [1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [6, 2], [7, 2], [7, 3], [7, 4], [6, 4], [6, 5], [5, 5], [5, 6], [5, 7], [6, 7], [6, 8], [7, 8], [7, 9], [8, 9], [9, 9]],
    },
    {
        difficulty: 5,
        landscape: 4,
        description: "Good conditioned route with good hill view. But many monkey.",
        route: [[0, 0], [1, 0], [1, 1], [1, 2], [1, 3], [0, 3], [0, 4], [0, 5], [1, 5], [2, 5], [3, 5], [3, 6], [3, 7], [2, 7], [2, 8], [2, 9]],
    },
    {
        difficulty: 4,
        landscape: 5,
        description: "Valuable hill & sea view, high difficulty, but it worth!",
        route: [[0, 0], [1, 0], [1, 1], [1, 2], [1, 3], [0, 3], [0, 4], [0, 5], [1, 5], [2, 5], [3, 5], [4, 5], [5, 5], [5, 6], [5, 7], [6, 7], [6, 8], [7, 8], [7, 9], [8, 9], [9, 9]],
    },
    {
        difficulty: 3,
        landscape: 1,
        description: "This is a challenging trial with deep slope.",
        route: [[0, 0], [1, 0], [1, 1], [1, 2], [1, 3], [0, 3], [0, 4], [0, 5], [1, 5], [2, 5], [3, 5], [4, 5], [5, 5], [6, 5], [6, 4], [7, 4],[7, 3], [7, 2], [8, 2], [8, 1], [9, 1]],
    },
]

const difficulty = [0, 1, 2, 3, 4, 5];
const landscape = [0, 1, 2, 3, 4, 5];
const tags = ["sea", "hill", "river", "plant", "animal", "monkey", "deep", "beautiful"];

class Data {

    constructor(trails, rememberLen) {
        this.trails = trails;
        this.rememberLen = rememberLen;
        this.routesIndices = [];
        this.trails.forEach((trail, i) => {
            trail.route.forEach((p, i) => {
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
    static preprocessText(text){
        if(text)
            return text.replace(/[^a-zA-Z ]/g, "").toLowerCase();
        return "";
    }
    static encodeTags(ts) {
        let t = new Array(tags.length).fill(0);
        ts.forEach(tag => {
            if (tag)
                t[tags.indexOf(tag)] = 1;
        });
        return t;
    }
    static textToTags(text){
        return Data.encodeTags(tags.map(tag => Data.preprocessText(text).indexOf(tag) >= 0 ? tag : null));
    }
    async prepareData(examplePerRoute) {
        let input = new tf.TensorBuffer([
            this.trails.length * examplePerRoute, this.rememberLen, this.pointLen]);
        let label = new tf.TensorBuffer([this.trails.length * examplePerRoute, this.pointLen]);

        let difficultyInput = [];
        let landscapeInput = [];
        let tagsInput = [];

        this.trails.forEach((trail, ti) => {
            let randomList = [];
            for (let i = 0;
                i < trail.route.length - this.rememberLen;
                i++) {
                randomList.push(i);
            }
            tf.util.shuffle(randomList);
            for (let i = ti * examplePerRoute; i < ti * examplePerRoute + examplePerRoute; i++) {
                const routeStartIndex = randomList[(i % examplePerRoute) % randomList.length];
                for (let j = 0; j < this.rememberLen; j++) {
                    const routePointIndex = routeStartIndex + j;
                    console.log("i:" + i, "j:" + j, "route point index:" + routePointIndex, "route point:" + trail.route[routePointIndex]);
                    input.set(1, i, j, this.encode(Data.conv2Value(trail.route[routePointIndex])));
                }
                const targetPointIndex = routeStartIndex + this.rememberLen;
                console.log("i:" + i, "target point index:" + targetPointIndex, "target point:" + trail.route[targetPointIndex]);
                label.set(1, i, this.encode(Data.conv2Value(trail.route[targetPointIndex])));
                difficultyInput.push(trail.difficulty);
                landscapeInput.push(trail.landscape);
                tagsInput.push(Data.textToTags(trail.description));
            }
        });
        return { input: [input.toTensor(), tf.tensor2d(tagsInput), tf.tensor1d(difficultyInput), tf.tensor1d(landscapeInput)], label: label.toTensor() };
    }
}



module.exports = { map, trails, tags, difficulty, landscape, Data }