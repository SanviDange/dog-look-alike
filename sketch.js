// sketch.js

let faceMesh;
let capture;
let faces = [];

function setup() {
  let cnv = createCanvas(640, 480);
  let btn = document.getElementById('matchBtn');
  document.body.insertBefore(cnv.elt, btn);

  capture = createCapture(VIDEO);
  capture.size(640, 480);
  capture.hide();

  faceMesh = ml5.faceMesh({ maxFaces: 1 }, () => {
    console.log('FaceMesh ready');
  });

  faceMesh.detectStart(capture, (results) => {
    faces = results;
  });

  loadDogVectors();
}

function draw() {
  image(capture, 0, 0);
  if (faces.length > 0) {
    drawMesh(faces[0]);
  }
}

function drawMesh(face) {
  stroke(0, 255, 0);
  strokeWeight(1);
  noFill();
  for (let pt of face.keypoints) {
    point(pt.x, pt.y);
  }
}

// Average 30 frames to reduce expression noise
function matchMe() {
  if (faces.length === 0) {
    alert('No face detected yet — wait a moment and try again.');
    return;
  }

  let samples = [];
  let count = 0;

  let interval = setInterval(() => {
    if (faces.length > 0) {
      samples.push(extractRatios(faces[0].keypoints));
      count++;
    }
    if (count >= 30) {
      clearInterval(interval);

      let avgVector = samples[0].map((_, i) =>
        samples.reduce((sum, v) => sum + v[i], 0) / samples.length
      );

      console.log('Vector (18):', avgVector.map(n => n.toFixed(4)).join(', '));
      let matches = findTopMatches(avgVector);
      displayMatches(matches);
    }
  }, 50);
}

// 18 features — mirrors process.py exactly
function extractRatios(kp) {

  let faceLeft   = kp[234].x;
  let faceRight  = kp[454].x;
  let faceTop    = kp[10].y;
  let faceBottom = kp[152].y;
  let faceWidth  = faceRight - faceLeft;
  let faceHeight = faceBottom - faceTop;

  // Left eye
  let leftOuter  = kp[33];
  let leftInner  = kp[133];
  let leftTop    = kp[159];
  let leftBottom = kp[145];
  let leftEyeWidth    = dist(leftOuter.x, leftOuter.y, leftInner.x, leftInner.y);
  let leftEyeH        = dist(leftTop.x, leftTop.y, leftBottom.x, leftBottom.y);
  let leftEyeOpenness = leftEyeH / (leftEyeWidth + 0.0001);

  // Right eye
  let rightOuter  = kp[362];
  let rightInner  = kp[263];
  let rightTop    = kp[386];
  let rightBottom = kp[374];
  let rightEyeWidth    = dist(rightOuter.x, rightOuter.y, rightInner.x, rightInner.y);
  let rightEyeH        = dist(rightTop.x, rightTop.y, rightBottom.x, rightBottom.y);
  let rightEyeOpenness = rightEyeH / (rightEyeWidth + 0.0001);

  // Eye centers
  let leftEyeCx  = (leftOuter.x  + leftInner.x)  / 2;
  let leftEyeCy  = (leftOuter.y  + leftInner.y)  / 2;
  let rightEyeCx = (rightOuter.x + rightInner.x) / 2;
  let rightEyeCy = (rightOuter.y + rightInner.y) / 2;

  let interocular = dist(leftEyeCx, leftEyeCy, rightEyeCx, rightEyeCy) / faceWidth;

  // Nose
  let noseLeft  = kp[49];
  let noseRight = kp[279];
  let noseTipX  = (noseLeft.x + noseRight.x) / 2;
  let noseTipY  = (noseLeft.y + noseRight.y) / 2;
  let noseWidth = dist(noseLeft.x, noseLeft.y, noseRight.x, noseRight.y) / faceWidth;

  // Mouth
  let mouthLeft  = kp[61];
  let mouthRight = kp[291];
  let mouthMidX  = (mouthLeft.x + mouthRight.x) / 2;
  let mouthMidY  = (mouthLeft.y + mouthRight.y) / 2;
  let mouthWidth = dist(mouthLeft.x, mouthLeft.y, mouthRight.x, mouthRight.y) / faceWidth;

  // Brows
  let leftBrow  = kp[105];
  let rightBrow = kp[334];
  let leftBrowHeight  = (leftBrow.y  - faceTop) / faceHeight;
  let rightBrowHeight = (rightBrow.y - faceTop) / faceHeight;
  let browWidth = dist(leftBrow.x, leftBrow.y, rightBrow.x, rightBrow.y) / faceWidth;

  // New structural features
  let noseToMouth = dist(noseTipX, noseTipY, mouthMidX, mouthMidY) / faceHeight;

  let eyesMidX = (leftEyeCx + rightEyeCx) / 2;
  let eyesMidY = (leftEyeCy + rightEyeCy) / 2;
  let eyeToNose = dist(eyesMidX, eyesMidY, noseTipX, noseTipY) / faceHeight;

  let mouthYRatio = (mouthMidY - faceTop) / faceHeight;
  let eyeYRatio   = (eyesMidY  - faceTop) / faceHeight;

  let noseToMouthWidth = noseWidth / (mouthWidth + 0.0001);

  let avgEyeWidth = (leftEyeWidth / faceWidth + rightEyeWidth / faceWidth) / 2;
  let eyeSpacingRatio = avgEyeWidth / (interocular + 0.0001);

  let browToEyeLeft  = Math.abs(leftBrow.y  - leftEyeCy)  / faceHeight;
  let browToEyeRight = Math.abs(rightBrow.y - rightEyeCy) / faceHeight;
  let browToEye = (browToEyeLeft + browToEyeRight) / 2;

  return [
    leftEyeWidth    / faceWidth,   // 0
    rightEyeWidth   / faceWidth,   // 1
    leftEyeOpenness,               // 2
    rightEyeOpenness,              // 3
    interocular,                   // 4
    noseWidth,                     // 5
    mouthWidth,                    // 6
    faceHeight / faceWidth,        // 7
    leftBrowHeight,                // 8
    rightBrowHeight,               // 9
    browWidth,                     // 10
    noseToMouth,                   // 11
    eyeToNose,                     // 12
    mouthYRatio,                   // 13
    eyeYRatio,                     // 14
    noseToMouthWidth,              // 15
    eyeSpacingRatio,               // 16
    browToEye,                     // 17
  ];
}

function displayMatches(matches) {
  let container = document.getElementById('results');
  container.innerHTML = '';
  matches.forEach((match, i) => {
    container.innerHTML += `
      <div class="match">
        <img src="images/${match.image}" />
        <p>#${i + 1} · ${match.distance.toFixed(3)}</p>
      </div>
    `;
  });
}