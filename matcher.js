// matcher.js

let dogVectors = [];

async function loadDogVectors() {
  const response = await fetch('dog_vectors.json');
  dogVectors = await response.json();
  console.log(`Loaded ${dogVectors.length} dogs`);
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
  let results = dogVectors.map(dog => ({
    image: dog.image,
    distance: euclideanDistance(humanVector, dog.vector)
  }));

  results.sort((a, b) => a.distance - b.distance);
  return results.slice(0, topN);
}