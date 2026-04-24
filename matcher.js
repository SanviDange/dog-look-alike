let dogVectors = [];
let dogMeans = null;
let dogStds = null;
let dogNormCache = null; // cached normalized dog vectors
let featureVariances = null; // track which features actually vary in dogs
let adaptiveWeights = null; // weights adjusted by feature importance

async function loadDogVectors() {
  const response = await fetch('dog_vectors.json');
  dogVectors = await response.json();
  console.log(`Loaded ${dogVectors.length} dogs`);
  // Precompute per-feature mean/std for z-scoring
  if (dogVectors.length > 0 && Array.isArray(dogVectors[0].vector)) {
    const m = dogVectors[0].vector.length;
    dogMeans = new Array(m).fill(0);
    dogStds  = new Array(m).fill(0);
    featureVariances = new Array(m).fill(0);

    // mean
    for (let v of dogVectors) {
      for (let i = 0; i < m; i++) dogMeans[i] += Number(v.vector[i]) || 0;
    }
    for (let i = 0; i < m; i++) dogMeans[i] /= dogVectors.length;

    // variance and std
    for (let v of dogVectors) {
      for (let i = 0; i < m; i++) {
        const d = (Number(v.vector[i]) || 0) - dogMeans[i];
        featureVariances[i] += d * d;
        dogStds[i] += d * d;
      }
    }
    for (let i = 0; i < m; i++) {
      featureVariances[i] = featureVariances[i] / dogVectors.length;
      dogStds[i] = Math.sqrt(dogStds[i] / dogVectors.length) || 0.0001;
    }

    // Compute adaptive weights: HIGH weight for features with HIGH variance (discriminative)
    // LOW weight for features with LOW variance (non-discriminative)
    computeAdaptiveWeights(featureVariances, m);

    // build normalized cache
    dogNormCache = dogVectors.map(d => ({ image: d.image, vector: normalizeWithStats(d.vector) }));
    console.log('Computed mean/std for', m, 'features');
    console.log('Feature variances:', featureVariances.map(v => v.toFixed(6)).join(' | '));
  }
}

function computeAdaptiveWeights(variances, numFeatures) {
  // Find max and min variance to normalize
  const maxVar = Math.max(...variances);
  const minVar = Math.min(...variances);
  const varRange = maxVar - minVar || 1;

  adaptiveWeights = new Array(numFeatures);
  
  for (let i = 0; i < numFeatures; i++) {
    // Normalize variance to 0-1, then map to weight range
    // High variance features get higher weight
    const normVar = (variances[i] - minVar) / varRange;
    
    // Static base weight for this feature
    const baseWeight = WEIGHTS[i] || 1.0;
    
    // Adaptive multiplier: features that vary a lot in dogs get HIGHER weight
    // Features that don't vary in dogs get LOWER weight
    const adaptiveMultiplier = 0.5 + (normVar * 1.5); // range: 0.5 to 2.0
    
    adaptiveWeights[i] = baseWeight * adaptiveMultiplier;
  }
  
  console.log('Adaptive weights computed based on dog dataset variance');
}

const WEIGHTS = [
  0.3, 0.3,  // 0-1: eye widths (STRUCTURAL - LOW)
  5.0, 5.0,  // 2-3: eye openness (EXPRESSION - VERY HIGH)
  0.2,       // 4: interocular (STRUCTURAL - VERY LOW, barely varies)
  0.3,       // 5: nose width (STRUCTURAL - LOW)
  0.3,       // 6: mouth width (STRUCTURAL - LOW)
  0.5,       // 7: face roundness (STRUCTURAL - LOW)
  0.3, 0.3,  // 8-9: brow heights (STRUCTURAL - LOW)
  0.3,       // 10: brow width (STRUCTURAL - LOW)
  0.3,       // 11: nose to mouth (STRUCTURAL - LOW)
  0.3,       // 12: eye to nose (STRUCTURAL - LOW)
  0.2,       // 13: mouth Y position (STRUCTURAL - VERY LOW)
  0.2,       // 14: eye Y position (STRUCTURAL - VERY LOW)
  0.3,       // 15: nose/mouth width ratio (STRUCTURAL - LOW)
  0.3,       // 16: eye spacing ratio (STRUCTURAL - LOW)
  0.3,       // 17: brow to eye (STRUCTURAL - LOW)
  2.0,       // 18: outer mouth openness (EXPRESSION - MODERATE)
  15.0,      // 19: INNER mouth opening (EXPRESSION - ULTRA HIGH - most discriminative for "mouth closed" vs "open") !!!
  8.0,       // 20: mouth corner raise (EXPRESSION - VERY HIGH)
  9.0,       // 21: brow raise (EXPRESSION - VERY HIGH)
  7.0,       // 22: L eye squeeze (EXPRESSION - VERY HIGH)
];

function weightedDistance(vecA, vecB, useAdaptive = false) {
  // Defensive: ensure both vectors are same length and numeric
  if (!Array.isArray(vecA) || !Array.isArray(vecB)) {
    console.warn('weightedDistance: non-array input', vecA, vecB);
    return Infinity;
  }
  if (vecA.length !== vecB.length) {
    console.warn('weightedDistance: vector length mismatch', vecA.length, vecB.length);
    return Infinity;
  }

  // Choose weights: adaptive (if computed) or static
  const weights = useAdaptive && adaptiveWeights ? adaptiveWeights : WEIGHTS;

  let sum = 0;
  for (let i = 0; i < vecA.length; i++) {
    let a = Number(vecA[i]);
    let b = Number(vecB[i]);
    let w = Number(weights[i]);

    if (!isFinite(a) || !isFinite(b) || !isFinite(w)) {
      // If any value isn't finite, bail out and treat as very far apart
      console.warn(`weightedDistance: non-finite component at index ${i}`, a, b, w);
      return Infinity;
    }

    let diff = (a - b) * w;
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

// Matching mode: 'weighted' (original), 'zscore' (z-scored), 'cosine' (direction-based)
// ALWAYS compute all 3 modes for comparison
window.MATCH_MODE = window.MATCH_MODE || 'weighted';

function findTopMatches(humanVector, topN = 1) {
  if (!Array.isArray(humanVector)) {
    console.warn('findTopMatches: humanVector is not an array', humanVector);
    return [];
  }


  // ===== MODE 1: WEIGHTED (original, raw features, ADAPTIVE WEIGHTS) =====
  let weightedResults = [];
  for (let dog of dogVectors) {
    let dist = Infinity;
    if (dog && Array.isArray(dog.vector)) dist = weightedDistance(humanVector, dog.vector, true); // useAdaptive=true
    weightedResults.push({ image: dog.image, distance: dist });
  }
  weightedResults.sort((a, b) => {
    if (!isFinite(a.distance) && !isFinite(b.distance)) return 0;
    if (!isFinite(a.distance)) return 1;
    if (!isFinite(b.distance)) return -1;
    return a.distance - b.distance;
  });
  const topWeighted = weightedResults.slice(0, topN);
  console.log(`WEIGHTED Top ${topN}: ${topWeighted.map((r, i) => `${i + 1}. ${r.image} (${r.distance.toFixed(4)})`).join(' | ')}`);

  // ===== MODE 2: ZSCORE (normalized Euclidean, ADAPTIVE WEIGHTS) =====
  let zscoreResults = [];
  if (dogMeans && dogStds && dogNormCache) {
    const hvZscore = normalizeWithStats(humanVector);
    for (let d of dogNormCache) {
      const dist = weightedDistance(hvZscore, d.vector, true); // useAdaptive=true
      zscoreResults.push({ image: d.image, distance: dist });
    }
    zscoreResults.sort((a, b) => {
      if (!isFinite(a.distance) && !isFinite(b.distance)) return 0;
      if (!isFinite(a.distance)) return 1;
      if (!isFinite(b.distance)) return -1;
      return a.distance - b.distance;
    });
  }
  const topZscore = zscoreResults.slice(0, topN);
  console.log(`ZSCORE  Top ${topN}: ${topZscore.map((r, i) => `${i + 1}. ${r.image} (${r.distance.toFixed(4)})`).join(' | ')}`);

  // ===== MODE 3: COSINE (direction/angle based) =====
  let cosineResults = [];
  if (dogMeans && dogStds && dogNormCache) {
    const hvCosine = normalizeWithStats(humanVector);
    for (let d of dogNormCache) {
      const sim = cosineSimilarity(hvCosine, d.vector);
      cosineResults.push({ image: d.image, distance: -sim }); // negative similarity so sort is ascending
    }
    cosineResults.sort((a, b) => {
      if (!isFinite(a.distance) && !isFinite(b.distance)) return 0;
      if (!isFinite(a.distance)) return 1;
      if (!isFinite(b.distance)) return -1;
      return a.distance - b.distance;
    });
  }
  const topCosine = cosineResults.slice(0, topN);
  console.log(`COSINE  Top ${topN}: ${topCosine.map((r, i) => `${i + 1}. ${r.image} (${(-r.distance).toFixed(4)})`).join(' | ')}`);

  // ===== RETURN SELECTED MODE =====
  let results = [];
  const mode = window.MATCH_MODE;
  if (mode === 'zscore') results = zscoreResults;
  else if (mode === 'cosine') results = cosineResults;
  else results = weightedResults; // default to weighted

  // Optional diversity filter: prefer different breed prefixes (before the underscore)
  if (window.DIVERSIFY) {
    const seen = new Set();
    const diverse = [];
    for (let r of results) {
      const prefix = (r.image || '').split('_')[0];
      if (!seen.has(prefix)) {
        seen.add(prefix);
        diverse.push(r);
      }
      if (diverse.length >= topN) break;
    }
    // If not enough diverse entries, fall back to top results
    if (diverse.length < topN) {
      for (let r of results) {
        if (diverse.indexOf(r) === -1) diverse.push(r);
        if (diverse.length >= topN) break;
      }
    }
    return diverse.slice(0, topN);
  }

  return results.slice(0, topN);
}

function normalizeWithStats(vec) {
  if (!dogMeans || !dogStds) return vec.slice();
  let out = new Array(vec.length);
  for (let i = 0; i < vec.length; i++) {
    const v = Number(vec[i]) || 0;
    const std = dogStds[i] > 1e-6 ? dogStds[i] : 1.0;
    out[i] = (v - dogMeans[i]) / std;
  }
  return out;
}

function cosineSimilarity(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = Number(a[i]) || 0;
    const y = Number(b[i]) || 0;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  if (na <= 0 || nb <= 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}