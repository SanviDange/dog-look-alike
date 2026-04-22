const fs = require('fs');

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
  if (!Array.isArray(vecA) || !Array.isArray(vecB) || vecA.length !== vecB.length) return Infinity;
  let sum = 0;
  for (let i = 0; i < vecA.length; i++) {
    let a = Number(vecA[i]), b = Number(vecB[i]), w = Number(WEIGHTS[i]);
    if (!isFinite(a) || !isFinite(b) || !isFinite(w)) return Infinity;
    let d = (a - b) * w;
    sum += d * d;
  }
  return Math.sqrt(sum);
}

let raw = fs.readFileSync('dog_vectors.json', 'utf8');
let dogs = JSON.parse(raw);
console.log('Loaded', dogs.length, 'dogs');

// compute mean/std
const m = dogs[0].vector.length;
let means = new Array(m).fill(0);
let stds = new Array(m).fill(0);
for (let d of dogs) for (let i = 0; i < m; i++) means[i] += Number(d.vector[i]) || 0;
for (let i = 0; i < m; i++) means[i] /= dogs.length;
for (let d of dogs) for (let i = 0; i < m; i++) { const delta = (Number(d.vector[i]) || 0) - means[i]; stds[i] += delta * delta; }
for (let i = 0; i < m; i++) stds[i] = Math.sqrt(stds[i] / dogs.length) || 0;

console.log('First 5 means:', means.slice(0,5));
console.log('First 5 stds :', stds.slice(0,5));

let base = dogs[0].vector;
let distances = dogs.slice(0, 50).map(d => ({ image: d.image, d: weightedDistance(base, d.vector) }));
console.log(distances.slice(0, 10));
let uniq = new Set(distances.map(x => x.d));
console.log('Unique distances in first 50:', uniq.size);
