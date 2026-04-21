// matcher.js

let dogVectors = [];

async function loadDogVectors() {
  const response = await fetch('dog_vectors.json');
  dogVectors = await response.json();
  console.log(`Loaded ${dogVectors.length} dogs`);
}

const WEIGHTS = [
  2.0, 2.0,  // eye widths
  1.5, 1.5,  // eye openness
  2.5,       // interocular
  2.0,       // nose width
  2.0,       // mouth width
  3.0,       // face roundness
  1.0, 1.0,  // brow heights
  1.5,       // brow width
  2.0,       // nose to mouth
  2.0,       // eye to nose
  1.5, 1.5,  // mouth/eye y position
  2.0,       // nose/mouth width ratio
  1.5,       // eye spacing
  1.0,       // brow to eye
  3.0,       // mouth open
  2.0,       // mouth corner raise
  2.0,       // brow raise
  1.5, 1.5,  // eye squeeze
];

function weightedDistance(vecA, vecB) {
  let sum = 0;
  for (let i = 0; i < vecA.length; i++) {
    let diff = (vecA[i] - vecB[i]) * WEIGHTS[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function findTopMatches(humanVector, topN = 3) {
  let results = dogVectors.map(dog => ({
    image: dog.image,
    distance: weightedDistance(humanVector, dog.vector)
  }));

  results.sort((a, b) => a.distance - b.distance);
  console.log('Top 3 images:', results.slice(0, 3).map(r => r.image).join(', '));
  return results.slice(0, topN);
}