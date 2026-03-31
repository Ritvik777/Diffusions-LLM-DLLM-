import { useState, useEffect, useRef, useCallback } from "react";

// ============================================================
// TEXT DIFFUSION MODEL — ALGORITHM DEMONSTRATION
// ============================================================
// This implements the core idea behind diffusion-based text
// generation (like Mercury/LLaDA):
//
// 1. FORWARD PROCESS: Corrupt text by replacing tokens with [MASK]
// 2. REVERSE PROCESS: Iteratively denoise by predicting masked tokens
//    with confidence scores, unmasking the most confident first
// 3. Unlike autoregressive LLMs (left-to-right, one token at a time),
//    diffusion models refine ALL positions simultaneously
//
// This is a simplified educational demo using a vocabulary-based
// probability model rather than a full neural network.
// ============================================================

// --- Vocabulary & "Learned" Distribution ---
// In a real dLLM, a transformer predicts token probabilities.
// Here we simulate this with topic-conditioned word distributions.

const TOPICS = {
  science: {
    label: "Science",
    seeds: [
      "The theory of relativity fundamentally changed our understanding of space and time in the universe",
      "Quantum mechanics describes the behavior of particles at the subatomic level with probability waves",
      "Neural networks learn complex patterns from data by adjusting connection weights through backpropagation",
      "DNA carries the genetic instructions used in the growth and development of all living organisms",
      "Black holes are regions of spacetime where gravity is so strong that nothing can escape them",
      "Evolution through natural selection drives the diversity of life across millions of years",
      "Photosynthesis converts sunlight into chemical energy that sustains nearly all life on earth",
      "The standard model of physics describes fundamental particles and three of four known forces",
    ],
    vocab: {
      high: ["the", "of", "and", "in", "is", "that", "a", "to", "are", "by", "from", "with", "for", "at", "on", "all", "can", "our", "its", "has", "an", "as", "into", "through", "where", "so", "how", "this", "these", "their", "be", "not", "or", "but", "which", "have", "been"],
      medium: ["theory", "quantum", "energy", "particles", "universe", "gravity", "light", "matter", "space", "time", "model", "system", "force", "field", "waves", "atoms", "mass", "data", "complex", "fundamental", "describes", "understanding", "behavior", "patterns", "regions", "structure", "process", "level", "known", "strong", "chemical", "across", "nearly", "living", "used"],
      low: ["relativity", "mechanics", "subatomic", "probability", "backpropagation", "photosynthesis", "evolution", "selection", "organisms", "genetic", "instructions", "networks", "spacetime", "diversity", "converts", "sunlight", "sustains", "adjusting", "connection", "weights", "changed", "nothing", "escape", "drives", "millions", "standard", "nuclear"],
    },
  },
  philosophy: {
    label: "Philosophy",
    seeds: [
      "Consciousness remains one of the deepest mysteries in philosophy and cognitive science today",
      "The nature of reality has been debated by thinkers across every civilization throughout history",
      "Free will and determinism present a fundamental paradox that challenges our sense of agency",
      "Ethics explores the boundaries between right and wrong in human conduct and moral reasoning",
      "Knowledge requires justified true belief according to the classical epistemological framework",
      "The meaning of existence cannot be separated from the subjective experience of being alive",
      "Language shapes thought and perception creating distinct worldviews across different cultures",
      "Truth is not simply correspondence with facts but involves coherence and pragmatic value",
    ],
    vocab: {
      high: ["the", "of", "and", "in", "is", "a", "that", "to", "has", "been", "by", "our", "not", "but", "with", "from", "between", "every", "across", "one", "be", "are", "or", "this", "its", "an", "as", "simply", "true", "can"],
      medium: ["consciousness", "reality", "knowledge", "truth", "meaning", "existence", "nature", "human", "thought", "reason", "moral", "sense", "experience", "belief", "value", "world", "mind", "freedom", "right", "wrong", "conduct", "perception", "distinct", "different", "fundamental", "classical", "subjective", "creating"],
      low: ["philosophy", "determinism", "epistemological", "paradox", "civilization", "coherence", "pragmatic", "correspondence", "worldviews", "mysteries", "cognitive", "boundaries", "challenges", "debated", "thinkers", "explores", "requires", "justified", "framework", "separated", "involves", "cultures", "deepest", "agency", "shapes"],
    },
  },
  code: {
    label: "Code & AI",
    seeds: [
      "Machine learning algorithms discover hidden patterns in large datasets through iterative optimization",
      "Transformers use attention mechanisms to process sequential data in parallel rather than sequentially",
      "Gradient descent minimizes the loss function by updating parameters in the direction of steepest decline",
      "Diffusion models generate high quality outputs by learning to reverse a gradual noising process",
      "Reinforcement learning agents maximize cumulative reward through trial and error in an environment",
      "Large language models predict the next token based on context from billions of training examples",
      "Convolutional neural networks extract hierarchical features from images using learned filter kernels",
      "Transfer learning enables models pretrained on large datasets to adapt quickly to new specific tasks",
    ],
    vocab: {
      high: ["the", "of", "and", "in", "to", "a", "by", "from", "on", "an", "through", "that", "is", "are", "use", "based", "rather", "than", "using", "for", "with", "new", "its", "can", "or", "as", "high"],
      medium: ["learning", "models", "data", "training", "function", "process", "parameters", "optimization", "patterns", "context", "token", "quality", "outputs", "features", "networks", "datasets", "agents", "direction", "examples", "parallel", "sequential", "hidden", "large", "neural", "gradual", "specific", "tasks", "quickly", "predict", "next"],
      low: ["machine", "algorithms", "transformers", "attention", "mechanisms", "gradient", "descent", "minimizes", "diffusion", "reinforcement", "cumulative", "reward", "convolutional", "hierarchical", "kernels", "pretrained", "transfer", "iterative", "noising", "steepest", "decline", "sequentially", "environment", "billions", "extract", "enables", "adapt", "discover", "updating", "reverse"],
    },
  },
};

const MASK_TOKEN = "[MASK]";

// --- Core Diffusion Algorithm ---

function tokenize(text) {
  return text.split(/\s+/).filter(Boolean);
}

function forwardDiffusion(tokens, noiseLevel) {
  // Forward process: corrupt tokens by replacing with [MASK]
  // Higher noise = more masks. At noise=1.0, everything is masked.
  return tokens.map((token) => {
    if (Math.random() < noiseLevel) return MASK_TOKEN;
    return token;
  });
}

function predictToken(position, context, topicVocab, originalTokens) {
  // Simulates what a neural network would do:
  // Given the current context (partially masked), predict the token
  // at `position` with a confidence score.
  //
  // In a real dLLM (Mercury), a transformer processes ALL positions
  // simultaneously and outputs probability distributions over the
  // vocabulary for each masked position.

  const original = originalTokens[position];
  const allWords = [
    ...topicVocab.high,
    ...topicVocab.medium,
    ...topicVocab.low,
  ];

  // Calculate how many unmasked neighbors exist (more context = higher confidence)
  let contextScore = 0;
  for (let i = Math.max(0, position - 3); i <= Math.min(context.length - 1, position + 3); i++) {
    if (i !== position && context[i] !== MASK_TOKEN) {
      contextScore += 1 / (1 + Math.abs(i - position));
    }
  }
  const contextRatio = contextScore / 3;

  // Simulate prediction: with some probability, predict correctly
  // More context = higher chance of correct prediction
  const correctProb = 0.3 + 0.6 * contextRatio;
  const confidence = 0.2 + 0.7 * contextRatio + 0.1 * Math.random();

  if (Math.random() < correctProb) {
    return { token: original, confidence };
  } else {
    // "Hallucinate" — pick a related word
    const pool = Math.random() < 0.6 ? topicVocab.medium : topicVocab.low;
    const predicted = pool[Math.floor(Math.random() * pool.length)];
    return { token: predicted, confidence: confidence * 0.6 };
  }
}

function denoisingStep(maskedTokens, originalTokens, topicVocab, unmaskRatio) {
  // REVERSE DIFFUSION STEP
  // 1. For ALL masked positions, predict tokens simultaneously
  // 2. Rank predictions by confidence
  // 3. Unmask the top `unmaskRatio` fraction of masked tokens
  //
  // This is the key difference from autoregressive models:
  // - AR: always unmask left-to-right
  // - Diffusion: unmask in ORDER OF CONFIDENCE (any position!)

  const predictions = [];
  const current = [...maskedTokens];

  // Step 1: Predict all masked positions in parallel
  for (let i = 0; i < current.length; i++) {
    if (current[i] === MASK_TOKEN) {
      const pred = predictToken(i, current, topicVocab, originalTokens);
      predictions.push({ index: i, ...pred });
    }
  }

  if (predictions.length === 0) return { tokens: current, unmasked: [], predictions: [] };

  // Step 2: Sort by confidence (highest first)
  predictions.sort((a, b) => b.confidence - a.confidence);

  // Step 3: Unmask top-k most confident predictions
  const numToUnmask = Math.max(1, Math.ceil(predictions.length * unmaskRatio));
  const unmasked = predictions.slice(0, numToUnmask);

  for (const pred of unmasked) {
    current[pred.index] = pred.token;
  }

  return { tokens: current, unmasked, predictions };
}

// --- Visualization Helpers ---

function getTokenColor(token, isJustUnmasked, confidence) {
  if (token === MASK_TOKEN) return { bg: "var(--mask-bg)", text: "var(--mask-text)", border: "var(--mask-border)" };
  if (isJustUnmasked) {
    const hue = Math.floor(confidence * 120); // red(0) -> green(120)
    return {
      bg: `hsla(${hue}, 80%, 45%, 0.15)`,
      text: `hsl(${hue}, 70%, 35%)`,
      border: `hsl(${hue}, 70%, 50%)`,
    };
  }
  return { bg: "var(--token-bg)", text: "var(--token-text)", border: "var(--token-border)" };
}

// --- Main Component ---

export default function TextDiffusionDemo() {
  const [topic, setTopic] = useState("science");
  const [numSteps, setNumSteps] = useState(8);
  const [isRunning, setIsRunning] = useState(false);
  const [history, setHistory] = useState([]);
  const [currentStep, setCurrentStep] = useState(-1);
  const [selectedSeed, setSelectedSeed] = useState(0);
  const [speed, setSpeed] = useState(800);
  const [showAlgorithm, setShowAlgorithm] = useState(false);
  const intervalRef = useRef(null);
  const historyEndRef = useRef(null);

  const topicData = TOPICS[topic];
  const originalText = topicData.seeds[selectedSeed];
  const originalTokens = tokenize(originalText);

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  useEffect(() => {
    if (historyEndRef.current) {
      historyEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [history]);

  const runDiffusion = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);

    const tokens = tokenize(originalText);
    // Start fully masked
    const fullyMasked = tokens.map(() => MASK_TOKEN);
    const steps = [
      {
        tokens: fullyMasked,
        unmasked: [],
        stepNum: 0,
        label: "Pure Noise (t=T)",
        maskedCount: tokens.length,
      },
    ];

    // Pre-compute all denoising steps
    let current = [...fullyMasked];
    for (let s = 1; s <= numSteps; s++) {
      const unmaskRatio = s < numSteps ? 1 / (numSteps - s + 1) : 1.0;
      const result = denoisingStep(current, tokens, topicData.vocab, unmaskRatio);
      current = result.tokens;
      const maskedCount = current.filter((t) => t === MASK_TOKEN).length;
      steps.push({
        tokens: [...current],
        unmasked: result.unmasked,
        predictions: result.predictions,
        stepNum: s,
        label: s === numSteps ? "Final Output (t=0)" : `Denoising Step ${s}/${numSteps}`,
        maskedCount,
      });
    }

    setHistory([steps[0]]);
    setCurrentStep(0);
    setIsRunning(true);

    let stepIdx = 1;
    intervalRef.current = setInterval(() => {
      if (stepIdx >= steps.length) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
        setIsRunning(false);
        return;
      }
      const nextStep = steps[stepIdx];
      if (!nextStep) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
        setIsRunning(false);
        return;
      }
      setHistory((prev) => [...prev, nextStep]);
      setCurrentStep(stepIdx);
      stepIdx++;
    }, speed);
  }, [originalText, numSteps, topicData, speed]);

  const reset = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setHistory([]);
    setCurrentStep(-1);
    setIsRunning(false);
  };

  const unmaskedIndicesInStep = (step) => {
    if (!step || !step.unmasked) return new Set();
    return new Set(step.unmasked.map((u) => u.index));
  };

  const confidenceForIndex = (step, idx) => {
    if (!step || !step.unmasked) return 0;
    const found = step.unmasked.find((u) => u.index === idx);
    return found ? found.confidence : 0;
  };

  return (
    <div style={{
      fontFamily: "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace",
      background: "#0a0a0f",
      color: "#e0e0e8",
      minHeight: "100vh",
      padding: "0",
      overflow: "auto",
      ["--mask-bg"]: "rgba(255, 60, 100, 0.12)",
      ["--mask-text"]: "#ff3c64",
      ["--mask-border"]: "#ff3c6440",
      ["--token-bg"]: "rgba(120, 200, 255, 0.08)",
      ["--token-text"]: "#94d0f0",
      ["--token-border"]: "rgba(120, 200, 255, 0.15)",
      ["--accent"]: "#00e5a0",
      ["--accent-dim"]: "#00e5a030",
      ["--surface"]: "#12121a",
      ["--surface-2"]: "#1a1a25",
      ["--border"]: "#2a2a3a",
    }}>
      {/* Header */}
      <div style={{
        borderBottom: "1px solid var(--border)",
        padding: "24px 32px",
        background: "linear-gradient(180deg, #0f0f18 0%, #0a0a0f 100%)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "8px" }}>
          <div style={{
            width: "10px", height: "10px", borderRadius: "50%",
            background: "var(--accent)", boxShadow: "0 0 12px var(--accent)",
          }} />
          <h1 style={{
            margin: 0, fontSize: "20px", fontWeight: 700,
            letterSpacing: "-0.5px", color: "#fff",
          }}>
            Text Diffusion Algorithm
          </h1>
          <span style={{
            fontSize: "11px", padding: "3px 10px", borderRadius: "20px",
            background: "var(--accent-dim)", color: "var(--accent)",
            fontWeight: 600, letterSpacing: "0.5px",
          }}>
            dLLM DEMO
          </span>
        </div>
        <p style={{
          margin: 0, fontSize: "13px", color: "#666680",
          maxWidth: "700px", lineHeight: 1.5,
        }}>
          Unlike autoregressive LLMs that generate left→right, diffusion models start from
          pure noise and iteratively denoise — unmasking tokens by <em style={{ color: "#888" }}>confidence</em>, not position.
        </p>
      </div>

      <div style={{ padding: "24px 32px", maxWidth: "960px", margin: "0 auto" }}>
        {/* Controls */}
        <div style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "16px",
          marginBottom: "24px",
        }}>
          {/* Topic */}
          <div style={{ background: "var(--surface)", borderRadius: "10px", padding: "16px", border: "1px solid var(--border)" }}>
            <label style={{ fontSize: "10px", textTransform: "uppercase", letterSpacing: "1.5px", color: "#555570", display: "block", marginBottom: "10px", fontWeight: 700 }}>
              Topic Domain
            </label>
            <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
              {Object.entries(TOPICS).map(([key, val]) => (
                <button key={key} onClick={() => { setTopic(key); setSelectedSeed(0); reset(); }}
                  style={{
                    padding: "6px 14px", borderRadius: "6px", border: "1px solid",
                    borderColor: topic === key ? "var(--accent)" : "var(--border)",
                    background: topic === key ? "var(--accent-dim)" : "transparent",
                    color: topic === key ? "var(--accent)" : "#888",
                    cursor: "pointer", fontSize: "12px", fontFamily: "inherit", fontWeight: 600,
                    transition: "all 0.2s",
                  }}>
                  {val.label}
                </button>
              ))}
            </div>
          </div>

          {/* Params */}
          <div style={{ background: "var(--surface)", borderRadius: "10px", padding: "16px", border: "1px solid var(--border)" }}>
            <label style={{ fontSize: "10px", textTransform: "uppercase", letterSpacing: "1.5px", color: "#555570", display: "block", marginBottom: "10px", fontWeight: 700 }}>
              Parameters
            </label>
            <div style={{ display: "flex", gap: "16px", alignItems: "center" }}>
              <div>
                <span style={{ fontSize: "11px", color: "#666" }}>Steps: </span>
                <select value={numSteps} onChange={(e) => { setNumSteps(Number(e.target.value)); reset(); }}
                  disabled={isRunning}
                  style={{
                    background: "var(--surface-2)", color: "#ccc", border: "1px solid var(--border)",
                    borderRadius: "4px", padding: "4px 8px", fontFamily: "inherit", fontSize: "12px",
                  }}>
                  {[4, 6, 8, 10, 12, 16].map((n) => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
              </div>
              <div>
                <span style={{ fontSize: "11px", color: "#666" }}>Speed: </span>
                <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}
                  disabled={isRunning}
                  style={{
                    background: "var(--surface-2)", color: "#ccc", border: "1px solid var(--border)",
                    borderRadius: "4px", padding: "4px 8px", fontFamily: "inherit", fontSize: "12px",
                  }}>
                  <option value={400}>Fast</option>
                  <option value={800}>Normal</option>
                  <option value={1400}>Slow</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Seed selector */}
        <div style={{
          background: "var(--surface)", borderRadius: "10px", padding: "16px",
          border: "1px solid var(--border)", marginBottom: "24px",
        }}>
          <label style={{ fontSize: "10px", textTransform: "uppercase", letterSpacing: "1.5px", color: "#555570", display: "block", marginBottom: "10px", fontWeight: 700 }}>
            Target Text (Ground Truth)
          </label>
          <div style={{ display: "flex", gap: "6px", flexWrap: "wrap", marginBottom: "12px" }}>
            {topicData.seeds.map((_, i) => (
              <button key={i} onClick={() => { setSelectedSeed(i); reset(); }}
                disabled={isRunning}
                style={{
                  width: "28px", height: "28px", borderRadius: "6px",
                  border: "1px solid",
                  borderColor: selectedSeed === i ? "var(--accent)" : "var(--border)",
                  background: selectedSeed === i ? "var(--accent-dim)" : "transparent",
                  color: selectedSeed === i ? "var(--accent)" : "#666",
                  cursor: isRunning ? "not-allowed" : "pointer",
                  fontSize: "12px", fontFamily: "inherit", fontWeight: 700,
                }}>
                {i + 1}
              </button>
            ))}
          </div>
          <div style={{
            padding: "12px 16px", background: "#0e0e16", borderRadius: "8px",
            fontSize: "13px", lineHeight: 1.7, color: "#78c8ff", letterSpacing: "0.2px",
            border: "1px solid rgba(120, 200, 255, 0.1)",
          }}>
            {originalText}
          </div>
        </div>

        {/* Run Button */}
        <div style={{ display: "flex", gap: "12px", marginBottom: "24px" }}>
          <button onClick={isRunning ? reset : runDiffusion}
            style={{
              padding: "12px 32px", borderRadius: "8px", border: "none",
              background: isRunning
                ? "linear-gradient(135deg, #ff3c64, #ff6b3c)"
                : "linear-gradient(135deg, #00e5a0, #00c8ff)",
              color: "#000", fontFamily: "inherit", fontSize: "13px",
              fontWeight: 700, cursor: "pointer", letterSpacing: "0.5px",
              transition: "all 0.3s",
              boxShadow: isRunning
                ? "0 4px 20px rgba(255, 60, 100, 0.3)"
                : "0 4px 20px rgba(0, 229, 160, 0.3)",
            }}>
            {isRunning ? "⏹ STOP" : "▶ RUN DIFFUSION"}
          </button>
          <button onClick={() => setShowAlgorithm(!showAlgorithm)}
            style={{
              padding: "12px 24px", borderRadius: "8px",
              border: "1px solid var(--border)", background: "var(--surface)",
              color: "#888", fontFamily: "inherit", fontSize: "13px",
              fontWeight: 600, cursor: "pointer",
            }}>
            {showAlgorithm ? "Hide" : "Show"} Algorithm
          </button>
        </div>

        {/* Algorithm Explanation */}
        {showAlgorithm && (
          <div style={{
            background: "var(--surface)", borderRadius: "10px", padding: "20px",
            border: "1px solid var(--border)", marginBottom: "24px",
            fontSize: "12px", lineHeight: 1.8, color: "#999",
          }}>
            <div style={{ color: "var(--accent)", fontWeight: 700, fontSize: "13px", marginBottom: "12px" }}>
              Core Algorithm: Masked Diffusion for Text
            </div>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", color: "#bbb" }}>
              <div style={{ color: "#ff6b9d" }}>{"// Forward Process (Training)"}</div>
              <div style={{ paddingLeft: "16px", marginBottom: "8px" }}>
                <div>x₀ = original_text.tokenize()</div>
                <div><span style={{ color: "#666" }}>for</span> t = 1 <span style={{ color: "#666" }}>to</span> T:</div>
                <div style={{ paddingLeft: "16px" }}>xₜ = mask(xₜ₋₁, noise_schedule[t])</div>
                <div style={{ paddingLeft: "16px", color: "#666" }}>// progressively more tokens → [MASK]</div>
              </div>
              <div style={{ color: "#ff6b9d" }}>{"// Reverse Process (Generation)"}</div>
              <div style={{ paddingLeft: "16px", marginBottom: "8px" }}>
                <div>x_T = all_masks <span style={{ color: "#666" }}>// pure noise</span></div>
                <div><span style={{ color: "#666" }}>for</span> t = T <span style={{ color: "#666" }}>to</span> 1:</div>
                <div style={{ paddingLeft: "16px" }}>predictions = model(xₜ) <span style={{ color: "#666" }}>// predict ALL masked positions</span></div>
                <div style={{ paddingLeft: "16px" }}>ranked = sort_by_confidence(predictions)</div>
                <div style={{ paddingLeft: "16px" }}>xₜ₋₁ = unmask_top_k(xₜ, ranked) <span style={{ color: "#666" }}>// reveal most confident first</span></div>
              </div>
              <div style={{ color: "#00e5a0", marginTop: "8px" }}>
                {"// KEY INSIGHT: tokens are unmasked by CONFIDENCE, not by position."}
              </div>
              <div style={{ color: "#00e5a0" }}>
                {"// Function words ('the','of') are easy → unmasked early."}
              </div>
              <div style={{ color: "#00e5a0" }}>
                {"// Domain-specific words → unmasked last (need more context)."}
              </div>
            </div>
          </div>
        )}

        {/* Diffusion Steps Visualization */}
        {history.length > 0 && (
          <div style={{ marginBottom: "24px" }}>
            <div style={{
              fontSize: "10px", textTransform: "uppercase", letterSpacing: "1.5px",
              color: "#555570", marginBottom: "16px", fontWeight: 700,
            }}>
              Reverse Diffusion Process — t=T → t=0
            </div>

            {history.map((step, si) => {
              if (!step || !step.tokens) return null;
              const justUnmasked = unmaskedIndicesInStep(step);
              const progress = (step.maskedCount ?? 0) / originalTokens.length;

              return (
                <div key={si} style={{
                  marginBottom: "12px",
                  opacity: si <= currentStep ? 1 : 0.3,
                  transition: "opacity 0.5s",
                  animation: si === currentStep ? "fadeSlideIn 0.4s ease-out" : undefined,
                }}>
                  {/* Step header */}
                  <div style={{
                    display: "flex", alignItems: "center", gap: "12px",
                    marginBottom: "8px",
                  }}>
                    <span style={{
                      fontSize: "10px", fontWeight: 700, color: "var(--accent)",
                      minWidth: "120px",
                    }}>
                      {step.label}
                    </span>
                    {/* Progress bar */}
                    <div style={{
                      flex: 1, height: "3px", background: "var(--border)",
                      borderRadius: "2px", overflow: "hidden",
                    }}>
                      <div style={{
                        width: `${(1 - progress) * 100}%`, height: "100%",
                        background: "linear-gradient(90deg, var(--accent), #00c8ff)",
                        transition: "width 0.5s ease-out",
                        borderRadius: "2px",
                      }} />
                    </div>
                    <span style={{ fontSize: "10px", color: "#555", minWidth: "90px", textAlign: "right" }}>
                      {step.maskedCount} / {originalTokens.length} masked
                    </span>
                  </div>

                  {/* Token grid */}
                  <div style={{
                    display: "flex", flexWrap: "wrap", gap: "4px",
                    padding: "12px", background: "var(--surface)",
                    borderRadius: "8px", border: "1px solid var(--border)",
                  }}>
                    {step.tokens.map((token, ti) => {
                      const isNew = justUnmasked.has(ti);
                      const conf = isNew ? confidenceForIndex(step, ti) : 0;
                      const colors = getTokenColor(token, isNew, conf);

                      return (
                        <span key={ti} title={isNew ? `confidence: ${(conf * 100).toFixed(0)}%` : undefined}
                          style={{
                            display: "inline-block",
                            padding: "3px 8px",
                            borderRadius: "4px",
                            fontSize: "12px",
                            fontWeight: token === MASK_TOKEN ? 400 : isNew ? 700 : 500,
                            background: colors.bg,
                            color: colors.text,
                            border: `1px solid ${colors.border}`,
                            transition: "all 0.3s",
                            animation: isNew ? "tokenReveal 0.5s ease-out" : undefined,
                            cursor: isNew ? "help" : "default",
                            letterSpacing: token === MASK_TOKEN ? "1px" : "0",
                          }}>
                          {token}
                        </span>
                      );
                    })}
                  </div>
                </div>
              );
            })}
            <div ref={historyEndRef} />
          </div>
        )}

        {/* Final comparison */}
        {!isRunning && history.length > 0 && currentStep >= numSteps && history[history.length - 1]?.tokens && (
          <div style={{
            background: "var(--surface)", borderRadius: "10px", padding: "20px",
            border: "1px solid rgba(0, 229, 160, 0.2)", marginBottom: "24px",
          }}>
            <div style={{
              fontSize: "10px", textTransform: "uppercase", letterSpacing: "1.5px",
              color: "var(--accent)", marginBottom: "12px", fontWeight: 700,
            }}>
              Comparison: Generated vs Original
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
              <div>
                <div style={{ fontSize: "10px", color: "#666", marginBottom: "6px", textTransform: "uppercase", letterSpacing: "1px" }}>Generated</div>
                <div style={{ fontSize: "13px", lineHeight: 1.7, color: "#e0e0e8" }}>
                  {history[history.length - 1].tokens.join(" ")}
                </div>
              </div>
              <div>
                <div style={{ fontSize: "10px", color: "#666", marginBottom: "6px", textTransform: "uppercase", letterSpacing: "1px" }}>Original</div>
                <div style={{ fontSize: "13px", lineHeight: 1.7, color: "#78c8ff" }}>
                  {originalText}
                </div>
              </div>
            </div>
            {(() => {
              const gen = history[history.length - 1].tokens;
              const matches = gen.filter((t, i) => t === originalTokens[i]).length;
              const accuracy = ((matches / originalTokens.length) * 100).toFixed(1);
              return (
                <div style={{
                  marginTop: "12px", padding: "8px 12px", background: "#0e0e16",
                  borderRadius: "6px", fontSize: "12px", color: "#888",
                }}>
                  Token accuracy: <span style={{ color: "var(--accent)", fontWeight: 700 }}>{accuracy}%</span>
                  {" "}({matches}/{originalTokens.length} exact matches)
                </div>
              );
            })()}
          </div>
        )}

        {/* Legend */}
        <div style={{
          display: "flex", gap: "20px", fontSize: "11px", color: "#555570",
          padding: "16px 0", borderTop: "1px solid var(--border)",
          flexWrap: "wrap",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{
              display: "inline-block", padding: "2px 8px", borderRadius: "3px",
              background: "var(--mask-bg)", color: "var(--mask-text)", fontSize: "10px", fontWeight: 600,
            }}>[MASK]</span>
            Masked / Noised
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{
              display: "inline-block", padding: "2px 8px", borderRadius: "3px",
              background: "hsla(90, 80%, 45%, 0.15)", color: "hsl(90, 70%, 35%)",
              border: "1px solid hsl(90, 70%, 50%)", fontSize: "10px", fontWeight: 600,
            }}>high</span>
            High confidence reveal
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{
              display: "inline-block", padding: "2px 8px", borderRadius: "3px",
              background: "hsla(20, 80%, 45%, 0.15)", color: "hsl(20, 70%, 35%)",
              border: "1px solid hsl(20, 70%, 50%)", fontSize: "10px", fontWeight: 600,
            }}>low</span>
            Low confidence reveal
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{
              display: "inline-block", padding: "2px 8px", borderRadius: "3px",
              background: "var(--token-bg)", color: "var(--token-text)",
              border: "1px solid var(--token-border)", fontSize: "10px", fontWeight: 600,
            }}>settled</span>
            Previously unmasked
          </div>
        </div>

        {/* Footer note */}
        <div style={{
          fontSize: "11px", color: "#444460", lineHeight: 1.7,
          padding: "16px 0",
        }}>
          <strong style={{ color: "#666" }}>Note:</strong> This demo uses a vocabulary-based probability
          model to simulate the denoising process. A production dLLM like Mercury uses a full
          transformer neural network trained on billions of tokens. The core algorithm — masked
          diffusion with confidence-ranked unmasking — is the same.
        </div>
      </div>

      <style>{`
        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes tokenReveal {
          0% { transform: scale(0.8); opacity: 0; }
          50% { transform: scale(1.1); }
          100% { transform: scale(1); opacity: 1; }
        }
        select:focus, button:focus {
          outline: 1px solid var(--accent);
          outline-offset: 1px;
        }
        * { box-sizing: border-box; }
      `}</style>
    </div>
  );
}
