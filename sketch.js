// sketch.js

let faceMesh;
let capture;
let faces = [];

function setup() {
  let cnv = createCanvas(640, 480);
  cnv.parent('canvas-container');

  capture = createCapture(VIDEO);
  capture.size(640, 480);
  capture.hide();

  // Start detection only after model is ready
  faceMesh = ml5.faceMesh({ maxFaces: 1 }, () => {
    console.log('FaceMesh ready');
    try {
      faceMesh.detectStart(capture, (results) => {
        faces = results;
      });
    } catch (err) {
      console.warn('detectStart failed, will retry once model is fully initialized', err);
      const t = setInterval(() => {
        try {
          faceMesh.detectStart(capture, (results) => { faces = results; });
          clearInterval(t);
          console.log('detectStart successful after retry');
        } catch (e) {
          // keep waiting
        }
      }, 200);
    }
  });

  loadDogVectors();
}

function matchMe() {
  if (faces.length === 0) {
    alert('No face detected yet — wait a moment and try again.');
    return;
  }

  let samples = [];
  let count = 0;
  const SAMPLE_COUNT = window.SAMPLE_COUNT || 30; // reduced to 30 frames for faster, less-smoothed capture

  let interval = setInterval(() => {
    if (faces.length > 0) {
      samples.push(extractRatios(faces[0].keypoints));
      count++;
    }
    if (count >= SAMPLE_COUNT) {
      clearInterval(interval);

      // compute per-component aggregates across samples
      const m = samples[0].length;
      let mins = new Array(m).fill(Infinity);
      let maxs = new Array(m).fill(-Infinity);

      // define which indices are expression-sensitive and which are structural
      // expressionIndices: eye openness (2,3), mouth openness (18,19), mouthCornerRaise (20), browRaise (21), eye squeeze (22)
      const expressionIndices = new Set([2, 3, 18, 19, 20, 21, 22]);
      const structuralIndices = [];
      for (let i = 0; i < m; i++) if (!expressionIndices.has(i)) structuralIndices.push(i);

      // collect columns
      const cols = [];
      for (let i = 0; i < m; i++) cols[i] = samples.map(s => s[i]);

      // structural: mean, expression: 75th percentile to preserve peaks
      function percentile(arr, p) {
        if (!arr || arr.length === 0) return 0;
        const sorted = arr.slice().sort((a,b) => a-b);
        const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * p)));
        return sorted[idx];
      }

      let agg = new Array(m).fill(0);
      for (let i of structuralIndices) {
        const col = cols[i];
        const mean = col.reduce((a,b)=>a+b,0)/col.length;
        agg[i] = mean;
        mins[i] = Math.min(...col);
        maxs[i] = Math.max(...col);
      }
      for (let i of Array.from(expressionIndices)) {
        const col = cols[i];
        // use 75th percentile (0.75) so we capture expressive peaks without noise
        const p75 = percentile(col, 0.75);
        agg[i] = p75;
        mins[i] = Math.min(...col);
        maxs[i] = Math.max(...col);
      }

      console.log('Vector (23) [agg]:', agg.map(n => Number(n).toFixed(4)).join(', '));
      console.log('Component ranges (min..max):', mins.map((mi, i) => `${mi.toFixed(4)}..${maxs[i].toFixed(4)}`).join(' | '));

      // Log sample variation (first 5 components)
      console.log('=== SAMPLE VARIATION ===');
      for (let i = 0; i < Math.min(m, 5); i++) {
        const col = cols[i];
        const meanVal = col.reduce((sum, x) => sum + x, 0) / col.length;
        const variance = col.reduce((sum, x) => sum + (x - meanVal) ** 2, 0) / col.length;
        console.log(`  Component ${i}: mean=${meanVal.toFixed(4)}, variance=${variance.toFixed(6)}, range=${mins[i].toFixed(4)}..${maxs[i].toFixed(4)}`);
      }

      let matches = findTopMatches(agg);
      displayMatches(matches);
    }
  }, 50);
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

function extractRatios(kp) {

  // Face bounding box
  let faceLeft   = kp[234].x;
  let faceRight  = kp[454].x;
  let faceTop    = kp[10].y;
  let faceBottom = kp[152].y;
  let faceWidth  = faceRight - faceLeft;
  let faceHeight = faceBottom - faceTop;
  
  // DEBUG: First call only, log face dimensions
  if (!window._loggedFaceOnce) {
    console.log(`Face dims: width=${faceWidth.toFixed(2)}, height=${faceHeight.toFixed(2)}, left=${faceLeft.toFixed(2)}, top=${faceTop.toFixed(2)}`);
    console.log(`Sample keypoints: kp[33]=${JSON.stringify(kp[33])}, kp[61]=${JSON.stringify(kp[61])}, kp[0]=${JSON.stringify(kp[0])}`);
    window._loggedFaceOnce = true;
  }

  // ── LEFT EYE ─────────────────────────────────────────────────
  let leftOuter  = kp[33];
  let leftInner  = kp[133];
  let leftTop    = kp[159];
  let leftBottom = kp[145];
  let leftEyeWidth    = dist(leftOuter.x, leftOuter.y, leftInner.x, leftInner.y);
  let leftEyeH        = dist(leftTop.x, leftTop.y, leftBottom.x, leftBottom.y);
  let leftEyeOpenness = leftEyeH / (leftEyeWidth + 0.0001);

  // ── RIGHT EYE ────────────────────────────────────────────────
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

  // ── NOSE ─────────────────────────────────────────────────────
  let noseLeft  = kp[49];
  let noseRight = kp[279];
  let noseTipX  = (noseLeft.x + noseRight.x) / 2;
  let noseTipY  = (noseLeft.y + noseRight.y) / 2;
  let noseWidth = dist(noseLeft.x, noseLeft.y, noseRight.x, noseRight.y) / faceWidth;

  // ── MOUTH ────────────────────────────────────────────────────
  // NOTE: Using FaceMesh landmarks. Approximate mapping to DogFLW:
  // DogFLW 38/41 ≈ ml5 61/291 (mouth corners)
  // DogFLW 45 (bottom lip) ≈ ml5 17 (lower lip center)
  let mouthLeft  = kp[61];
  let mouthRight = kp[291];
  let mouthWidth = dist(mouthLeft.x, mouthLeft.y, mouthRight.x, mouthRight.y) / faceWidth;
  
  // Mouth center: landmarks 61=left, 291=right (corners)
  let mouthMidX = (mouthLeft.x + mouthRight.x) / 2;
  let mouthMidY = (mouthLeft.y + mouthRight.y) / 2;

  // ── BROWS ────────────────────────────────────────────────────
  let leftBrow  = kp[105];
  let rightBrow = kp[334];
  let leftBrowHeight  = (leftBrow.y  - faceTop) / faceHeight;
  let rightBrowHeight = (rightBrow.y - faceTop) / faceHeight;
  let browWidth = dist(leftBrow.x, leftBrow.y, rightBrow.x, rightBrow.y) / faceWidth;

  // ── STRUCTURAL DERIVED ───────────────────────────────────────
  let noseToMouth = dist(noseTipX, noseTipY, mouthMidX, mouthMidY) / faceHeight;
  let eyesMidX    = (leftEyeCx + rightEyeCx) / 2;
  let eyesMidY    = (leftEyeCy + rightEyeCy) / 2;
  let eyeToNose   = dist(eyesMidX, eyesMidY, noseTipX, noseTipY) / faceHeight;
  let mouthYRatio = (mouthMidY - faceTop) / faceHeight;
  let eyeYRatio   = (eyesMidY  - faceTop) / faceHeight;

  let noseToMouthWidth = noseWidth / (mouthWidth + 0.0001);
  let avgEyeWidth      = (leftEyeWidth / faceWidth + rightEyeWidth / faceWidth) / 2;
  let eyeSpacingRatio  = avgEyeWidth / (interocular + 0.0001);

  let browToEyeLeft  = Math.abs(leftBrow.y  - leftEyeCy)  / faceHeight;
  let browToEyeRight = Math.abs(rightBrow.y - rightEyeCy) / faceHeight;
  let browToEye      = (browToEyeLeft + browToEyeRight) / 2;

  // ── EXPRESSION FEATURES ──────────────────────────────────────

  // Mouth openness — use INNER mouth landmarks that actually move with jaw
  // ml5: kp[13]=upper inner lip, kp[14]=lower inner lip (these move when mouth opens!)
  // PRIMARY: inner mouth opening (distance between upper and lower inner lip)
  let innerMouthOpen = 0;
  if (kp[13] && kp[14]) {
    innerMouthOpen = dist(kp[13].x, kp[13].y, kp[14].x, kp[14].y) / faceHeight;
  } else {
    innerMouthOpen = 0.01; // fallback
  }

  // Outer mouth opening — distance between outer lip landmarks
  // kp[0]=upper lip center, kp[17]=lower lip center (less motion)
  let mouthOpen = dist(kp[0].x, kp[0].y, kp[17].x, kp[17].y) / faceHeight;

  // Mouth corner raise — corners Y vs center of lips Y (smile detection)
  // DogFLW: (mouth_center_y - (landmarks[39][1] + landmarks[40][1]) / 2) / face_height
  // ml5: (mouthMidY - average corner Y) / faceHeight
  let cornerAvgY = (mouthLeft.y + mouthRight.y) / 2;
  let mouthCornerRaise = (mouthMidY - cornerAvgY) / faceHeight;

  // Brow raise — how far brows sit above eye centers
  let leftBrowRaise  = (leftEyeCy  - leftBrow.y)  / faceHeight;
  let rightBrowRaise = (rightEyeCy - rightBrow.y) / faceHeight;
  let browRaise      = (leftBrowRaise + rightBrowRaise) / 2;

  // Eye squeeze — raw eye height / face height
  let leftEyeSqueeze  = leftEyeH  / faceHeight;
  let rightEyeSqueeze = rightEyeH / faceHeight;

  // ── 23-FEATURE VECTOR ────────────────────────────────────────
  return [
    leftEyeWidth    / faceWidth,   // 0  L eye width / face width
    rightEyeWidth   / faceWidth,   // 1  R eye width / face width
    leftEyeOpenness,               // 2  L eye openness
    rightEyeOpenness,              // 3  R eye openness
    interocular,                   // 4  interocular / face width
    noseWidth,                     // 5  nose width / face width
    mouthWidth,                    // 6  mouth width / face width
    faceHeight / faceWidth,        // 7  face roundness
    leftBrowHeight,                // 8  L brow height / face height
    rightBrowHeight,               // 9  R brow height / face height
    browWidth,                     // 10 brow width / face width
    noseToMouth,                   // 11 nose-to-mouth / face height
    eyeToNose,                     // 12 eye-to-nose / face height
    mouthYRatio,                   // 13 mouth Y pos / face height
    eyeYRatio,                     // 14 eye Y pos / face height
    noseToMouthWidth,              // 15 nose width / mouth width
    eyeSpacingRatio,               // 16 avg eye width / interocular
    browToEye,                     // 17 brow-to-eye gap / face height
    mouthOpen,                     // 18 outer mouth openness
    innerMouthOpen,                // 19 inner mouth opening (more sensitive)
    mouthCornerRaise,              // 20 mouth corner raise
    browRaise,                     // 21 brow raise / face height
    leftEyeSqueeze,                // 22 L eye squeeze / face height
  ];
}

function displayMatches(matches) {
  let box = document.getElementById('dog-result-box');
  box.innerHTML = '';
  if (matches.length > 0) {
    box.innerHTML = `<img src="images/${matches[0].image}" style="width:100%; height:100%; object-fit:cover; border-radius:18px;" />`;
  }
}

// Diagnostic function to inspect face detection
function diagnoseFaceData() {
  if (faces.length === 0) {
    console.log('❌ NO FACE DETECTED');
    return;
  }
  
  const kp = faces[0].keypoints;
  console.log('=== FACE DIAGNOSIS ===');
  console.log(`Total keypoints: ${kp.length}`);
  console.log(`Confidence: ${faces[0].confidence || 'N/A'}`);
  
  // Sample a few key points to verify detection is working
  console.log('Sample keypoints:');
  console.log(`  kp[0] (lip top): (${kp[0].x.toFixed(1)}, ${kp[0].y.toFixed(1)})`);
  console.log(`  kp[10] (face top): (${kp[10].x.toFixed(1)}, ${kp[10].y.toFixed(1)})`);
  console.log(`  kp[33] (L eye outer): (${kp[33].x.toFixed(1)}, ${kp[33].y.toFixed(1)})`);
  console.log(`  kp[61] (mouth left): (${kp[61].x.toFixed(1)}, ${kp[61].y.toFixed(1)})`);
  console.log(`  kp[152] (face bottom): (${kp[152].x.toFixed(1)}, ${kp[152].y.toFixed(1)})`);
  
  // Extract one sample to see feature values
  const features = extractRatios(kp);
  console.log('Extracted features (should vary per face):');
  const names = [
    'L_eye_width', 'R_eye_width', 'L_eye_open', 'R_eye_open', 'interocular',
    'nose_width', 'mouth_width', 'face_roundness', 'L_brow_height', 'R_brow_height',
    'brow_width', 'nose_to_mouth', 'eye_to_nose', 'mouth_Y', 'eye_Y',
    'nose_to_mouth_width', 'eye_spacing', 'brow_to_eye', 'mouth_open', 'inner_mouth',
    'mouth_corner_raise', 'brow_raise', 'L_eye_squeeze'
  ];
  names.forEach((name, i) => {
    console.log(`  ${i.toString().padStart(2, ' ')}. ${name.padEnd(20, ' ')}: ${features[i].toFixed(4)}`);
  });
}