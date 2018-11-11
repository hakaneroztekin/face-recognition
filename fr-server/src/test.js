const fr = require('face-recognition');
const fs = require('fs');

const recognizer = fr.FaceRecognizer();
const detector = fr.FaceDetector();

const modelState = require('../model.json');

const content = fs.readFileSync(__dirname + "/test.txt", 'utf8');


const lines = content.split('\n').map(line => line.split('\t'));
const numberOfSets = Number(lines[0][0]);
const numberOfPairs = Number(lines[0][1]);
const total = numberOfPairs * numberOfSets * 2;

let numberOfCorrect = 0;
let errors = 0;

recognizer.load(modelState);

for(let i = 0; i < numberOfSets; i++) {
    console.log("---------> SET", i + 1);

    for(let j = 0; j < numberOfPairs; j++) {
        const [name, n1, n2] = lines[(i * 2 * numberOfPairs) + 1 + j];
        const imagePath1 = __dirname + "/lfw/" + name + "/" + name + "_" + n1.padStart(4, '0') + ".jpg";
        const imagePath2 = __dirname + "/lfw/" + name + "/" + name + "_" + n2.padStart(4, '0') + ".jpg";
        let image1, image2;

        try {
            image1 = fr.loadImage(imagePath1);
        } catch (err) {
            console.error("Couldn't open file", imagePath1);
            errors++;
            continue;
        }

        try {
            image2 = fr.loadImage(imagePath2);
        } catch (err) {
            console.error("Couldn't open file", imagePath2);
            errors++;
            continue;
        }

        const faceImages1 = detector.detectFaces(image1);

        if(faceImages1.length < 0) {
            console.error("No face is found!", name, n1);
            errors++;
            continue;
        };

        const faceImages2 = detector.detectFaces(image2);

        if(faceImages2.length < 0) {
            console.error("No face is found!", name, n2);
            errors++;
            continue;
        };
        
        const faceImage1 = faceImages1[0];
        const faceImage2 = faceImages2[0];
        
        const prediction1 = recognizer.predictBest(faceImage1); 
        const prediction2 = recognizer.predictBest(faceImage2); 

        if(prediction1.className === prediction2.className) {
            numberOfCorrect++;
        } else {
            console.log("Missmatched.", name, n1, n2);
        }
    }

    console.log("---------> FINISHED HALF");

    for(let j = 0; j < numberOfPairs; j++) {
        const [name1, n1, name2, n2] = lines[(i * 2 * numberOfPairs) + 1 + j + numberOfPairs];
        const imagePath1 = __dirname + "/lfw/" + name1 + "/" + name1 + "_" + n1.padStart(4, '0') + ".jpg";
        const imagePath2 = __dirname + "/lfw/" + name2 + "/" + name2 + "_" + n2.padStart(4, '0') + ".jpg";
        let image1, image2;

        try {
            image1 = fr.loadImage(imagePath1);
        } catch (err) {
            console.error("Couldn't open file", imagePath1);
            errors++;
            continue;
        }

        try {
            image2 = fr.loadImage(imagePath2);
        } catch (err) {
            console.error("Couldn't open file", imagePath2);
            errors++;
            continue;
        }
        
        const faceImages1 = detector.detectFaces(image1);

        if(faceImages1.length < 0) {
            console.error("No face is found!", name1, n1);
            errors++;
            continue;
        };

        const faceImages2 = detector.detectFaces(image2);

        if(faceImages2.length < 0) {
            console.error("No face is found!", name2, n2);
            errors++;
            continue;
        };
        
        const faceImage1 = faceImages1[0];
        const faceImage2 = faceImages2[0];
        
        const prediction1 = recognizer.predictBest(faceImage1); 
        const prediction2 = recognizer.predictBest(faceImage2); 

        if(prediction1.className !== prediction2.className) {
            numberOfCorrect++;
        } else {
            console.log("Matched.", name1, n1, name2, n2);
        }
    }
}

console.log("----------> RESULT: %" + (numberOfCorrect / (total - errors)) * 100, "Errors:" + errors);
