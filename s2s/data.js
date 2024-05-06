const fs = require('node:fs');

const tf = require('@tensorflow/tfjs-node');
//const w2v = require('word2vec');
//const Segmenter = require('node-analyzer');
//const segmenter = new Segmenter();

const trails = [
    {
        difficulty: 3,//1,
        landscape: 3,
        description: `位於香港仔郊野公園內有香港仔上下水塘，後者建於1890年，本為大成紙廠私人興建，至1929年政府為增加香港仔和鴨脷洲一帶的供水量，向紙廠購買水塘，再在其上游興建香港仔上水塘，並在1932年完成兩個水塘的修築及重建工程。水塘內的林道寬闊，從漁光道巴士站下車，沿香港仔水塘道上坡，不用多時已可看見橫臥於山林中的上水塘水壩。之後可一邊欣賞下水塘風景，一邊沿林蔭路進發，便可到達終點上水塘水壩，探索已列為法定古蹟的石橋、水掣房和水壩。香港仔水塘林道沿途會經過香港仔傷健樂園，裡面有小食亭、無障礙燒烤場及洗手間等設施，可作為補給站或午餐的好地方。
        注意事項:
        1) 由漁光村海鷗樓巴士站至香港仔郊野公園閘口 (郊野公園界外)路段比較斜，建議活動能力受障人士要有同行者陪同或直接乘的士到閘口出發。
        2) 郊野公園內斜度為約1:8-1:5 (7-10 度)。
        3) 沿途有野豬出沒，請勿餵飼。一般情況下，野豬都會避開遊人。但當野豬受挑釁或受驚後有可能會作出攻擊行為，特別是帶有幼豬的母豬及成年雄豬。
        4) 折返點為香港仔上水塘石橋尾。
        5) 如需為電動輪椅充電，可於辦公時間到香港仔樹木廊或郊野公園管理站向漁護署職員尋求協助。
        6) 其他注意事項及經驗分享：綠洲/ Trailwatch /Wheel Power Challenge`,
        route: [[0, 0], [1, 0], [1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2], [8, 1], [9, 1]],
    },
    {
        difficulty: 3,//2,
        landscape: 3,//5,
        description: "香港仔自然教育徑初段環繞香港仔下水塘而行，平坦易走。香港仔下水塘原是一家紙廠的私人水塘，其後為配合香港仔河谷供水計劃而被當時政府接管及改建，並於1932年重新啟用。走上自然教育徑，沿途除可遠眺香港仔避風塘的怡人景色外，亦能窺看融合中國宮廷和意大利的建築風格的天主教修道院，更有機會見到不少有趣的動植物，如作為「廿四味」成份之一的淡竹葉、用作包糭子的水銀竹，以及遨翔天際的黑鳶等。自然教育徑末段可找到俗稱「天花墩」的舊式量雨器，以及已被列為法定古蹟的上水塘水壩，絕對不可錯過呢！",
        route: [[0, 0], [1, 0], [1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [6, 2], [7, 2], [7, 3], [7, 4], [6, 4], [6, 5], [5, 5], [5, 6], [5, 7], [6, 7], [6, 8], [7, 8], [7, 9], [8, 9], [9, 9]],
    },
    {
        difficulty: 3,//5,
        landscape: 3,//4,
        description: "Good conditioned route with good hill view. But many monkey.",
        route: [[0, 0], [1, 0], [1, 1], [1, 2], [1, 3], [0, 3], [0, 4], [0, 5], [1, 5], [2, 5], [3, 5], [3, 6], [3, 7], [2, 7], [2, 8], [2, 9]],
    },
    {
        difficulty: 3,//4,
        landscape: 3,//5,
        description: "Valuable hill & sea view, high difficulty, but it worth!",
        route: [[0, 0], [1, 0], [1, 1], [1, 2], [1, 3], [0, 3], [0, 4], [0, 5], [1, 5], [2, 5], [3, 5], [4, 5], [5, 5], [5, 6], [5, 7], [6, 7], [6, 8], [7, 8], [7, 9], [8, 9], [9, 9]],
    },
    {
        difficulty: 3,
        landscape: 3,//1,
        description: "This is a challenging trial with deep slope.",
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
            this.trails.length/* * examplePerRoute*/, this.textMaxSize, this.vocab.length]);
        let decorderInput = new tf.TensorBuffer([
            this.trails.length/* * examplePerRoute*/, this.maxLength/*this.rememberSize*/, this.pointSize]);
        let target = new tf.TensorBuffer([this.trails.length/* * examplePerRoute*/, this.maxLength/*this.rememberSize*/, this.pointSize]);

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

            words.forEach((w, j) => {
                const index = this.vocab.indexOf(w.toLowerCase());
                encorderInput.set(1, i, j, index < 0 ? 0 : index);
            });

            for (let i = 0; i < trail.route.length; i++) {
                decorderInput.set(1, ti, i, this.encode(Data.conv2Value(trail.route[i])));
                if (i > 0) {
                    target.set(1, ti, i - 1, this.encode(Data.conv2Value(targetRoute[i])));
                }
                console.log(ti, i, targetRoute[i], Data.conv2Value(targetRoute[i]), this.encode(Data.conv2Value(targetRoute[i])));
            }

            /*let randomList = [];
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
            }*/
        });
        console.log("encorderInput", encorderInput.toTensor().arraySync());
        console.log("decorderInput", decorderInput.toTensor().arraySync());
        console.log("target", target.toTensor().arraySync());
        return { input: [encorderInput.toTensor(), decorderInput.toTensor()], target: target.toTensor() };
    }
}



module.exports = { trails, Data }