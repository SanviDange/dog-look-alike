// matcher.js
let dogVectors = [];
let _mean = [];
let _std = [];

async function loadDogVectors() {
  const response = await fetch('dog_vectors.json');
  dogVectors = await response.json();
  console.log(`Loaded ${dogVectors.length} dogs`);

  // Compute per-feature mean and std across ALL dogs.
  // This is the key fix: human face proportions cluster in a tiny region
  // of the raw feature space, so everyone matches the same "average" dogs.
  // Z-scoring stretches each feature relative to the dog distribution,
  // so small human differences map to large spread across the dog dataset.
  const N = dogVectors.length;
  const D = dogVectors[0].vector.length;
  _mean = new Array(D).fill(0);
  _std  = new Array(D).fill(1);

  for (let fi = 0; fi < D; fi++) {
    let sum = 0;
    for (let i = 0; i < N; i++) sum += dogVectors[i].vector[fi];
    _mean[fi] = sum / N;

    let variance = 0;
    for (let i = 0; i < N; i++) {
      const d = dogVectors[i].vector[fi] - _mean[fi];
      variance += d * d;
    }
    _std[fi] = Math.sqrt(variance / N) || 1; // fallback 1 for zero-variance features
  }

  console.log('Z-score stats ready');
}

function _zscore(vec) {
  return vec.map((v, i) => (v - _mean[i]) / _std[i]);
}

const WEIGHTS = [
  4.0, 4.0,  //  0-1  eye width
  0.5, 0.5,  //  2-3  eye openness (expression)
  4.0,       //  4    interocular
  4.0,       //  5    nose width
  0.5,       //  6    mouth width (expression)
  3.0,       //  7    face roundness
  2.5, 2.5,  //  8-9  brow heights
  2.5,       // 10    brow width
  2.0,       // 11    nose to mouth
  2.0,       // 12    eye to nose
  1.0, 1.0,  // 13-14 mouth/eye Y position
  3.0,       // 15    nose/mouth width ratio
  2.0,       // 16    eye spacing ratio
  2.0,       // 17    brow to eye gap
  0.3,       // 18    mouth openness (expression)
  0.3,       // 19    mouth corner raise (expression)
  1.5,       // 20    brow raise
  0.5, 0.5,  // 21-22 eye squeeze (expression)
];

function weightedDistance(zA, zB) {
  let sum = 0;
  for (let i = 0; i < zA.length; i++) {
    const diff = (zA[i] - zB[i]) * WEIGHTS[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function findTopMatches(humanVector, topN = 3) {
  const zHuman = _zscore(humanVector);

  let results = dogVectors.map(dog => ({
    image: dog.image,
    distance: weightedDistance(zHuman, _zscore(dog.vector))
  }));

  results.sort((a, b) => a.distance - b.distance);
  console.log('Top 3 images:', results.slice(0, 3).map(r => r.image).join(', '));
  return results.slice(0, topN);
}