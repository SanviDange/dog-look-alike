// matcher.js

let dogVectors = [];
let dogMeans = [];
let dogStds = [];

async function loadDogVectors() {
  const response = await fetch('dog_vectors.json');
  dogVectors = await response.json();
  console.log(`Loaded ${dogVectors.length} dogs`);
  computeNormStats();
}

// Compute mean and std for each of the 10 features across all dogs
function computeNormStats() {
  const n = dogVectors.length;
  const dim = dogVectors[0].vector.length;

  dogMeans = new Array(dim).fill(0);
  dogStds  = new Array(dim).fill(0);

  // Step 1: compute mean per feature
  for (let dog of dogVectors) {
    for (let i = 0; i < dim; i++) {
      dogMeans[i] += dog.vector[i];
    }
  }
  for (let i = 0; i < dim; i++) {
    dogMeans[i] /= n;
  }

  // Step 2: compute std per feature
  for (let dog of dogVectors) {
    for (let i = 0; i < dim; i++) {
      dogStds[i] += Math.pow(dog.vector[i] - dogMeans[i], 2);
    }
  }
  for (let i = 0; i < dim; i++) {
    dogStds[i] = Math.sqrt(dogStds[i] / n);
    if (dogStds[i] === 0) dogStds[i] = 1; // avoid divide by zero
  }

  console.log('Means:', dogMeans.map(n => n.toFixed(4)));
  console.log('Stds: ', dogStds.map(n => n.toFixed(4)));
}

// Normalize a vector using the dog population's mean and std
function normalize(vec) {
  return vec.map((v, i) => (v - dogMeans[i]) / dogStds[i]);
}

function euclideanDistance(vecA, vecB) {
  let sum = 0;
  for (let i = 0; i < vecA.length; i++) {
    let diff = vecA[i] - vecB[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function findTopMatches(humanVector, topN = 3) {
  // Normalize the human vector using dog population stats
  const normHuman = normalize(humanVector);

  let results = dogVectors.map(dog => ({
    image: dog.image,
    distance: euclideanDistance(normHuman, normalize(dog.vector))
  }));

  results.sort((a, b) => a.distance - b.distance);
  return results.slice(0, topN);
}