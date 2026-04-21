// sketch.js

let faceMesh;
let capture;
let faces = [];

function setup() {
  let cnv = createCanvas(640, 480);
  cnv.parent(document.body);         // attach to body (before button)
  cnv.elt.id = 'canvas';

  // Move canvas to the right place in the DOM
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

function matchMe() {
  if (faces.length === 0) { alert('No face detected'); return; }

  let kp = faces[0].keypoints;
  let humanVector = extractRatios(kp);
  
  console.log('vector:', humanVector.map(n => n.toFixed(4)).join(', '));

  let matches = findTopMatches(humanVector);
  displayMatches(matches);
}

function extractRatios(kp) {
  // Face bounding box
  let faceLeft   = kp[234].x;
  let faceRight  = kp[454].x;
  let faceTop    = kp[10].y;
  let faceBottom = kp[152].y;
  let faceWidth  = faceRight - faceLeft;
  let faceHeight = faceBottom - faceTop;

  // Left eye
  let leftEyeOuter  = kp[33];
  let leftEyeInner  = kp[133];
  let leftEyeTop    = kp[159];
  let leftEyeBottom = kp[145];
  let leftEyeWidth  = dist(leftEyeOuter.x, leftEyeOuter.y, leftEyeInner.x, leftEyeInner.y);
  let leftEyeOpenness = dist(leftEyeTop.x, leftEyeTop.y, leftEyeBottom.x, leftEyeBottom.y) / leftEyeWidth;

  // Right eye
  let rightEyeOuter  = kp[362];
  let rightEyeInner  = kp[263];
  let rightEyeTop    = kp[386];
  let rightEyeBottom = kp[374];
  let rightEyeWidth  = dist(rightEyeOuter.x, rightEyeOuter.y, rightEyeInner.x, rightEyeInner.y);
  let rightEyeOpenness = dist(rightEyeTop.x, rightEyeTop.y, rightEyeBottom.x, rightEyeBottom.y) / rightEyeWidth;

  // Interocular distance (inner corners)
  let interocular = dist(kp[133].x, kp[133].y, kp[362].x, kp[362].y);

  // Nose width (nostril tips)
  let noseWidth = dist(kp[49].x, kp[49].y, kp[279].x, kp[279].y);

  // Mouth width (corners)
  let mouthWidth = dist(kp[61].x, kp[61].y, kp[291].x, kp[291].y);

  // Brow heights (brow point to eye top, normalized by face height)
  let leftBrowHeight  = (kp[145].y - kp[105].y) / faceHeight;
  let rightBrowHeight = (kp[374].y - kp[334].y) / faceHeight;

  return [
    leftEyeWidth    / faceWidth,   // 1
    rightEyeWidth   / faceWidth,   // 2
    leftEyeOpenness,               // 3
    rightEyeOpenness,              // 4
    interocular     / faceWidth,   // 5
    noseWidth       / faceWidth,   // 6
    mouthWidth      / faceWidth,   // 7
    faceHeight      / faceWidth,   // 8
    leftBrowHeight,                // 9
    rightBrowHeight                // 10
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