const fs = require('node:fs');

const tf = require('@tensorflow/tfjs-node');
//const w2v = require('word2vec');
//const Segmenter = require('node-analyzer');
//const segmenter = new Segmenter();

const trails = [
    {
        difficulty: 3,//1,
        landscape: 3,
        description: `香港仔`,
        route: [[0, 0], [1, 0], [1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2], [8, 1], [9, 1]],
    },
    {
        difficulty: 3,//2,
        landscape: 3,//5,
        description: "水塘",
        route: [[0, 0], [1, 0], [1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [6, 2], [7, 2], [7, 3], [7, 4], [6, 4], [6, 5], [5, 5], [5, 6], [5, 7], [6, 7], [6, 8], [7, 8], [7, 9], [8, 9], [9, 9]],
    },
    {
        difficulty: 3,//5,
        landscape: 3,//4,
        description: "monkey",
        route: [[0, 0], [1, 0], [1, 1], [1, 2], [1, 3], [0, 3], [0, 4], [0, 5], [1, 5], [2, 5], [3, 5], [3, 6], [3, 7], [2, 7], [2, 8], [2, 9]],
    },
    {
        difficulty: 3,//4,
        landscape: 3,//5,
        description: "sea view",
        route: [[0, 0], [1, 0], [1, 1], [1, 2], [1, 3], [0, 3], [0, 4], [0, 5], [1, 5], [2, 5], [3, 5], [4, 5], [5, 5], [5, 6], [5, 7], [6, 7], [6, 8], [7, 8], [7, 9], [8, 9], [9, 9]],
    },
    {
        difficulty: 3,
        landscape: 3,//1,
        description: "challenging",
        route: [[0, 0], [1, 0], [1, 1], [1, 2], [1, 3], [0, 3], [0, 4], [0, 5], [1, 5], [2, 5], [3, 5], [4, 5], [5, 5], [6, 5], [6, 4], [7, 4], [7, 3], [7, 2], [8, 2], [8, 1], [9, 1]],
    },
];
/*let content = "";
trails.forEach((trail, ti) => {
    content += segmenter.analyze(trail.description);
});

try {
    console.log(content);
    fs.writeFileSync("s2s/desc.txt", content, { flag: 'wx' });
    // file written successfully
} catch (err) {
    console.error(err);
}*/
class Data {

    constructor(trails, textMaxSize, rememberSize) {
        this.textMaxSize = textMaxSize;
        this.rememberSize = rememberSize;
        this.trails = trails;
        this.routesIndices = [-1];
        this.maxLength = 0;
        this.trails.forEach((trail, i) => {
            if (trail.route.length > this.maxLength)
                this.maxLength = trail.route.length;
            trail.route.forEach((p, i) => {
                const v = Data.conv2Value(p);
                if (this.routesIndices.indexOf(v) == -1) this.routesIndices.push(v);
            });
        });
        this.maxLength+=1; //the end signal [-1]
        this.pointSize = this.routesIndices.length;
        //w2v.word2vec("s2s/desc.txt", "s2s/dict.txt", { size: 32, minCount: 1 });
        //w2v.loadModel("s2s/dict.txt", (error, m) => {
        //    this.w2vModel = m;
        //});

        try {
            const data = fs.readFileSync('s2s/desc.txt', 'utf8');
            this.vocab = data.toLowerCase().replaceAll("\n", " ").split(" ").filter(n => n);
        } catch (err) {
            console.error(err);
        }
        console.log("vocab length:"+this.vocab.length);
    }

    static conv2Coordinate(v) {
        if (v < 0)
            return [v];
        return [v % 10, Math.floor(v / 10)];
    }
    static conv2Value(c) {
        if (!c)
            return -1;
        if (c[0] < 0)
            return c[0];
        return (c[1] * 10) + c[0];
    }
    encode(v) {
        return this.routesIndices.indexOf(v);
    }
    decode(i) {
        return this.routesIndices[i];
    }
    async prepareData(examplePerRoute) {
        //let encorderInput = new tf.TensorBuffer([
        //    this.trails.length, this.textMaxSize, parseInt(this.w2vModel.size)]);
        let encorderInput = new tf.TensorBuffer([
            this.trails.length * examplePerRoute, this.textMaxSize, this.vocab.length]);
        let decorderInput = new tf.TensorBuffer([
            this.trails.length * examplePerRoute, this.maxLength/*this.rememberSize*/, this.pointSize]);
        let target = new tf.TensorBuffer([this.trails.length * examplePerRoute, this.maxLength/*this.rememberSize*/, this.pointSize]);

        this.trails.forEach((trail, ti) => {
            const targetRoute = trail.route.concat([[-1]]);

            //let descVec = new Array(parseInt(this.textMaxSize)).fill(new Array(parseInt(this.w2vModel.size)).fill(0));
            //descVec = Object.assign(descVec, this.w2vModel.getVectors(segmenter.analyze(trail.description)).map(v => v.values));
            //encorderInput.push(descVec);
            //this.w2vModel.getVectors(segmenter.analyze(trail.description)).map(v => v.values).forEach((v,i)=>{
            //    encorderInput.set(1, ti, i, v);
            //});


            //const words = segmenter.analyze(trail.description).split(" ");
            //const words = trail.description.split(" ");
            let words = [];
            this.vocab.forEach(v=>{
                if(trail.description.indexOf(v)>=0)
                    words.push(v);
            });
            //console.log(words);

            /*words.forEach((w, j) => {
                const index = this.vocab.indexOf(w.toLowerCase());
                encorderInput.set(1, ti, j, index < 0 ? 0 : index);
            });

            for (let i = 0; i < trail.route.length; i++) {
                decorderInput.set(1, ti, i, this.encode(Data.conv2Value(trail.route[i])));
                if (i > 0) {
                    target.set(1, ti, i - 1, this.encode(Data.conv2Value(targetRoute[i])));
                }
                console.log(ti, i, targetRoute[i], Data.conv2Value(targetRoute[i]), this.encode(Data.conv2Value(targetRoute[i])));
            }*/

            let randomList = [];
            for (let i = 0;
                i < trail.route.length;
                i++) {
                randomList.push(i);
            }
            tf.util.shuffle(randomList);

            for (let i = ti * examplePerRoute; i < ti * examplePerRoute + examplePerRoute; i++) {

                words.forEach((w, j) => {
                    const index = this.vocab.indexOf(w.toLowerCase());
                    encorderInput.set(1, i, j, index < 0 ? 0 : index);
                });
                const routeStartIndex = randomList[(i % examplePerRoute) % randomList.length];
                for (let j = 0; j < this.rememberSize; j++) {
                    const routePointIndex = routeStartIndex + j;
                    console.log("i:" + i, "j:" + j, "route point index:" + routePointIndex, "route point:" + targetRoute[routePointIndex]);
                    decorderInput.set(1, i, j, this.encode(Data.conv2Value(targetRoute[routePointIndex])));
                    if (j > 0) {
                        console.log("i:", i, "j:", (j - 1), "target point index:" + routePointIndex, "target point:" + targetRoute[routePointIndex]);
                        target.set(1, i, j - 1, this.encode(Data.conv2Value(targetRoute[routePointIndex])));
                    }
                }
            }
        });
        console.log("encorderInput", encorderInput.toTensor().arraySync());
        console.log("decorderInput", decorderInput.toTensor().arraySync());
        console.log("target", target.toTensor().arraySync());
        return { input: [encorderInput.toTensor(), decorderInput.toTensor()], target: target.toTensor() };
    }
}



module.exports = { trails, Data }