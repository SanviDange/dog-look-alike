/**
 * matcher.js — Fixed version
 *
 * Problems solved:
 *   1. Centroid dominance: raw weighted Euclidean favors dogs near the mean.
 *      Fix: z-score normalize every feature across the dog corpus first, so
 *      each dimension contributes equally before weights are applied.
 *   2. Dead features (feature 19 always 0): zero-variance features collapse
 *      to NaN during normalization and are automatically masked out.
 *   3. Diversity: top-N now uses max-distance filtering so results spread
 *      across the dog space rather than clustering around the same centroid dogs.
 */

let dogVectors = [];       // raw, as loaded from JSON
let dogStats   = [];       // per-feature { mean, std } across all dogs
let dogNormed  = [];       // z-scored dog vectors (same length as dogVectors)
let validFeatures = [];    // indices where std > 0  (dead features excluded)

// ─── Weights (applied AFTER z-score normalization) ───────────────────────────
// Indices match the 23 features described in the project spec.
const WEIGHTS = [
  2.0, 2.0,   //  0-1  eye widths
  1.5, 1.5,   //  2-3  eye openness
  2.5,        //  4    interocular
  2.0,        //  5    nose width
  2.0,        //  6    mouth width
  3.0,        //  7    face roundness
  1.0, 1.0,   //  8-9  brow heights
  1.5,        // 10    brow width
  2.0,        // 11    nose to mouth
  2.0,        // 12    eye to nose
  1.5, 1.5,   // 13-14 mouth/eye y position
  2.0,        // 15    nose/mouth width ratio
  1.5,        // 16    eye spacing
  1.0,        // 17    brow to eye
  3.0,        // 18    mouth open
  2.0,        // 19    mouth corner raise  ← often 0; masked if zero-variance
  2.0,        // 20    brow raise
  1.5, 1.5,   // 21-22 eye squeeze
];

// ─── Load + precompute ────────────────────────────────────────────────────────

async function loadDogVectors() {
  const response = await fetch('dog_vectors.json');
  dogVectors = await response.json();
  console.log(`Loaded ${dogVectors.length} dogs`);
  _precompute();
}

function _precompute() {
  const N = dogVectors.length;
  const D = dogVectors[0].vector.length;

  // 1. Compute per-feature mean and std across the entire dog corpus
  dogStats = Array.from({ length: D }, (_, fi) => {
    const vals = dogVectors.map(d => d.vector[fi]);
    const mean = vals.reduce((s, v) => s + v, 0) / N;
    const variance = vals.reduce((s, v) => s + (v - mean) ** 2, 0) / N;
    const std = Math.sqrt(variance);
    return { mean, std };
  });

  // 2. Identify valid (non-zero-variance) features
  validFeatures = dogStats
    .map((s, i) => (s.std > 1e-9 ? i : -1))
    .filter(i => i >= 0);

  const dead = D - validFeatures.length;
  if (dead > 0) {
    console.warn(`Matcher: ${dead} zero-variance feature(s) masked out:`,
      dogStats.map((s, i) => (s.std <= 1e-9 ? i : -1)).filter(i => i >= 0));
  }

  // 3. Z-score every dog vector (only valid features)
  dogNormed = dogVectors.map(dog => {
    const z = new Array(D).fill(0);
    for (const fi of validFeatures) {
      z[fi] = (dog.vector[fi] - dogStats[fi].mean) / dogStats[fi].std;
    }
    return { image: dog.image, vector: z };
  });

  console.log(`Matcher ready. Valid features: ${validFeatures.length}/${D}`);
}

// ─── Normalize a human vector into the same z-score space ────────────────────

function normalizeHumanVector(humanVec) {
  const D = humanVec.length;
  const z = new Array(D).fill(0);
  for (const fi of validFeatures) {
    z[fi] = (humanVec[fi] - dogStats[fi].mean) / dogStats[fi].std;
  }
  return z;
}

// ─── Distance (weighted Euclidean in z-score space) ──────────────────────────

function weightedDistance(zA, zB) {
  let sum = 0;
  for (const fi of validFeatures) {
    const diff = (zA[fi] - zB[fi]) * WEIGHTS[fi];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

// ─── Top-N with diversity enforcement ────────────────────────────────────────
/**
 * findTopMatches
 *   humanVector : raw 23-element array from sketch.js
 *   topN        : how many results to return (default 3)
 *   diversityRadius : min z-space distance between returned dogs (default 1.5)
 *                     Set to 0 to disable diversity filtering.
 */
function findTopMatches(humanVector, topN = 3, diversityRadius = 1.5) {
  if (dogNormed.length === 0) {
    console.error('Dog vectors not loaded yet.');
    return [];
  }

  const zHuman = normalizeHumanVector(humanVector);

  // Score all dogs
  const scored = dogNormed.map(dog => ({
    image: dog.image,
    vector: dog.vector,
    distance: weightedDistance(zHuman, dog.vector),
  }));
  scored.sort((a, b) => a.distance - b.distance);

  if (diversityRadius <= 0) {
    return scored.slice(0, topN).map(({ image, distance }) => ({ image, distance }));
  }

  // Greedy diversity filtering: pick next-closest dog that is at least
  // diversityRadius away (in z-space) from every already-selected dog.
  const selected = [];
  for (const candidate of scored) {
    if (selected.length >= topN) break;
    const tooClose = selected.some(
      s => weightedDistance(s.vector, candidate.vector) < diversityRadius
    );
    if (!tooClose) {
      selected.push(candidate);
    }
  }

  // Fallback: if diversity filter was too aggressive, fill with closest remaining
  if (selected.length < topN) {
    for (const candidate of scored) {
      if (selected.length >= topN) break;
      if (!selected.find(s => s.image === candidate.image)) {
        selected.push(candidate);
      }
    }
  }

  return selected.map(({ image, distance }) => ({ image, distance }));
}

// ─── Debug helper ─────────────────────────────────────────────────────────────
/**
 * debugVector(humanVector)
 * Call from the browser console to inspect which features are driving matches.
 * Prints a table of feature index, human z-score, and how that compares to
 * the winning dog.
 */
function debugVector(humanVector) {
  if (dogNormed.length === 0) { console.warn('Not loaded'); return; }
  const zHuman = normalizeHumanVector(humanVector);
  const top = findTopMatches(humanVector, 1, 0)[0];
  const winnerNormed = dogNormed.find(d => d.image === top.image);

  console.group(`Top match: ${top.image}  (dist=${top.distance.toFixed(4)})`);
  console.table(
    validFeatures.map(fi => ({
      feature: fi,
      human_z: zHuman[fi].toFixed(3),
      dog_z:   winnerNormed.vector[fi].toFixed(3),
      diff:    (zHuman[fi] - winnerNormed.vector[fi]).toFixed(3),
      weight:  WEIGHTS[fi],
      contribution: ((zHuman[fi] - winnerNormed.vector[fi]) * WEIGHTS[fi] ** 2).toFixed(4),
    }))
  );
  console.groupEnd();
}// matcher.js
let dogVectors = [];

// Per-feature stats computed once after loading
let _mean = [];
let _std  = [];

async function loadDogVectors() {
  const response = await fetch('dog_vectors.json');
  dogVectors = await response.json();
  console.log(`Loaded ${dogVectors.length} dogs`);
  _computeStats();
}

// Compute mean and std for each feature across all dogs
function _computeStats() {
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
      const diff = dogVectors[i].vector[fi] - _mean[fi];
      variance += diff * diff;
    }
    // Fallback to 1 if zero-variance (dead feature like mouth corner raise)
    // so it contributes nothing rather than dividing by zero
    _std[fi] = Math.sqrt(variance / N) || 1;
  }

  console.log('Feature stats computed. Dead features (std≈0):',
    _std.map((s, i) => s < 0.001 ? i : -1).filter(i => i >= 0));
}

// Normalize a raw vector into z-score space using dog corpus stats
function _normalize(vec) {
  return vec.map((v, i) => (v - _mean[i]) / _std[i]);
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
  // Normalize human + every dog into z-score space so no feature
  // dominates just because its raw range is larger than others.
  // Dead features (std≈0, e.g. feature 19) get std=1 fallback
  // so they contribute ~zero to distance automatically.
  const zHuman = _normalize(humanVector);

  let results = dogVectors.map(dog => ({
    image: dog.image,
    distance: weightedDistance(zHuman, _normalize(dog.vector))
  }));

  results.sort((a, b) => a.distance - b.distance);
  console.log('Top 3 images:', results.slice(0, 3).map(r => r.image).join(', '));
  return results.slice(0, topN);
}