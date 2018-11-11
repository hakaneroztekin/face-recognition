
// Dataset: http://vis-www.cs.umass.edu/lfw/lfw.tgz

const fr = require('face-recognition');
const fs = require('fs');

const detector = fr.FaceDetector();
const recognizer = fr.FaceRecognizer();

const arr = fs.readdirSync(__dirname + "/lfw")
const size = arr.length;
arr.forEach((file, index) => {

    const person = file;
    const faces = [];

    fs.readdirSync(__dirname + "/lfw/" + file).forEach(imageFile => {
        const image = fr.loadImage(__dirname + "/lfw/" + file + "/" + imageFile);
        const _faces = detector.detectFaces(image);
        if(!_faces.length) return;

        faces.push(_faces[0]);
    });

    recognizer.addFaces(faces, person);
});
    
const modelState = recognizer.serialize();
fs.writeFileSync('model.json', JSON.stringify(modelState));