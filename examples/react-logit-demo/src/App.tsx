import { useState } from "react";
import * as webllm from "@mlc-ai/web-llm";

// Easy to read text (predictable, simple grammar)
const EASY_TEXT =
  "The sun is shining brightly in the clear blue sky. Birds are singing in the trees, and the gentle wind blows through the leaves. It is a beautiful day to go for a walk in the park. Children are playing on the swings and slides, laughing with joy. A dog chases a ball across the green grass. Everyone seems happy and relaxed on this warm summer afternoon.";

// Difficult to read text (complex, technical, abstract)
const DIFFICULT_TEXT =
  "The epistemological foundations of the metaphysical dichotomy between subject and object necessitate a transcendental deduction of the categories of understanding, lest the manifold of intuition remain a chaotic aggregate of disparate sensations devoid of synthetic unity. This phenomenological reduction suspends the natural attitude, revealing the constitutive acts of consciousness that structure our experience of the lifeworld.";

interface TokenLogprob {
  token: string;
  logprob: number;
}

interface AnalysisResult {
  text: string;
  tokens: TokenLogprob[];
  stats: {
    min: number;
    max: number;
    avg: number;
    perplexity: number;
  } | null;
  modelName: string;
}

// Model Definitions
const AVAILABLE_MODELS = [
  {
    id: "TinyLlama",
    name: "TinyLlama 1.1B",
    description: "Lightweight, remote weights",
    config: {
      model:
        "https://huggingface.co/mlc-ai/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC",
      model_id: "TinyLlama",
      model_lib: "/TinyLlama-1.1B.wasm",
      vram_required_MB: 700,
      low_resource_required: true,
    },
  },
  {
    id: "Qwen2.5-1.5B",
    name: "Qwen2.5 1.5B",
    description: "High quality, local weights",
    config: {
      model: "/models/Qwen2.5-1.5B-q4f16_1",
      model_id: "Qwen2.5-1.5B",
      model_lib: "/models/Qwen2.5-1.5B-q4f16_1/Qwen2.5-1.5B.wasm",
      vram_required_MB: 1800,
      low_resource_required: false,
    },
  },
  {
    id: "Mistral-7B-v0.3",
    name: "Mistral 7B v0.3",
    description: "Highest quality, local weights, requires more VRAM",
    config: {
      model: "/models/Mistral-7B-v0.3-q4f16_1",
      model_id: "Mistral-7B-v0.3",
      model_lib: "/models/Mistral-7B-v0.3-q4f16_1/Mistral-7B-v0.3.wasm",
      vram_required_MB: 4600,
      low_resource_required: false,
      required_features: ["shader-f16"],
    },
  },
  // Add Mistral and Gemma here when they are ready/compiled
];

// Color scale from green (high prob) to red (low prob)
function getColorForLogprob(logprob: number): string {
  const normalized = Math.max(0, Math.min(1, (logprob + 10) / 10));
  const hue = normalized * 120;
  return "hsl(" + hue + ", 70%, 85%)";
}

function LogprobView({
  title,
  result,
  isLoading,
}: {
  title: string;
  result: AnalysisResult | null;
  isLoading: boolean;
}) {
  if (isLoading) {
    return <div style={styles.resultBoxEmpty}>Processing...</div>;
  }

  if (!result) {
    return <div style={styles.resultBoxEmpty}>Waiting for analysis...</div>;
  }

  return (
    <div style={styles.resultContainer}>
      <h3 style={{ margin: "0 0 10px 0", fontSize: 16 }}>
        {title}{" "}
        <span style={{ fontWeight: "normal", color: "#666" }}>
          ({result.modelName})
        </span>
      </h3>

      {result.stats && (
        <div style={styles.statsContainer}>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 12, color: "#666" }}>Perplexity</div>
            <div style={{ fontSize: 24, fontWeight: "bold", color: "#2196F3" }}>
              {result.stats.perplexity.toFixed(3)}
            </div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 12, color: "#666" }}>Avg Logprob</div>
            <div style={{ fontSize: 18, fontWeight: "bold", color: "#333" }}>
              {result.stats.avg.toFixed(3)}
            </div>
          </div>
        </div>
      )}

      <div style={styles.tokensContainer}>
        {result.tokens.map((tl, idx) => (
          <span
            key={idx}
            title={
              "token: " +
              tl.token +
              "\nlogprob: " +
              tl.logprob.toFixed(4) +
              "\nprob: " +
              (Math.exp(tl.logprob) * 100).toFixed(2) +
              "%"
            }
            style={{
              ...styles.token,
              background: getColorForLogprob(tl.logprob),
            }}
          >
            {tl.token}
          </span>
        ))}
      </div>
    </div>
  );
}

function App() {
  const [inputText, setInputText] = useState(EASY_TEXT);
  const [status, setStatus] = useState("Ready");
  const [progress, setProgress] = useState(0);
  const [engine, setEngine] = useState<webllm.MLCEngineInterface | null>(null);
  const [currentModelLoaded, setCurrentModelLoaded] = useState<string | null>(
    null,
  );

  const [leftModelId, setLeftModelId] = useState(AVAILABLE_MODELS[0].id);
  const [rightModelId, setRightModelId] = useState(
    AVAILABLE_MODELS[1]?.id || AVAILABLE_MODELS[0].id,
  );

  const [leftResult, setLeftResult] = useState<AnalysisResult | null>(null);
  const [rightResult, setRightResult] = useState<AnalysisResult | null>(null);

  const [processingSide, setProcessingSide] = useState<
    "left" | "right" | "both" | null
  >(null);

  const getEngineForModel = async (modelId: string) => {
    const modelDef = AVAILABLE_MODELS.find((m) => m.id === modelId);
    if (!modelDef)
      throw new Error(`Model configuration for ${modelId} not found`);

    if (engine && currentModelLoaded === modelId) {
      return engine;
    }

    setStatus(`Initializing ${modelDef.name}...`);
    // Need to load new model
    let eng = engine;
    if (!eng) {
      // First time
      eng = await webllm.CreateMLCEngine(modelDef.config.model_id, {
        appConfig: { model_list: [modelDef.config] },
        initProgressCallback: (report) => {
          setStatus(report.text);
          setProgress(report.progress * 100);
        },
      });
      setEngine(eng);
    } else {
      // Reload
      eng.setAppConfig({ model_list: [modelDef.config] });
      await eng.reload(modelDef.config.model_id);
    }

    setCurrentModelLoaded(modelId);
    return eng;
  };

  const analyze = async (
    targetText: string,
    modelId: string,
  ): Promise<AnalysisResult> => {
    const eng = await getEngineForModel(modelId);
    const modelDef = AVAILABLE_MODELS.find((m) => m.id === modelId)!;

    setStatus(`Running inference on ${modelDef.name}...`);

    const reply = await eng.chat.completions.create({
      messages: [{ role: "user", content: targetText }],
      max_tokens: 1,
      return_input_logprobs: true,
      logprobs: true,
      top_logprobs: 1,
    });

    const inputLogprobs = reply.input_logprobs;
    const tokens: TokenLogprob[] = [];

    if (inputLogprobs && Array.isArray(inputLogprobs)) {
      if (reply.input_tokens && Array.isArray(reply.input_tokens)) {
        for (
          let i = 0;
          i < reply.input_tokens.length && i < inputLogprobs.length;
          i++
        ) {
          tokens.push({
            token: String(reply.input_tokens[i]),
            logprob: inputLogprobs[i],
          });
        }
      }

      const validLogprobs = inputLogprobs.filter(
        (lp) => typeof lp === "number" && !isNaN(lp),
      ) as number[];
      let stats = null;
      if (validLogprobs.length > 0) {
        const avg =
          validLogprobs.reduce((a, b) => a + b, 0) / validLogprobs.length;
        stats = {
          min: Math.min(...validLogprobs),
          max: Math.max(...validLogprobs),
          avg: avg,
          perplexity: Math.exp(-avg),
        };
      }

      return { text: targetText, tokens, stats, modelName: modelDef.name };
    }
    throw new Error("No logprobs returned from model");
  };

  const runLeft = async () => {
    if (processingSide) return;
    setProcessingSide("left");
    try {
      const res = await analyze(inputText, leftModelId);
      setLeftResult(res);
      setStatus("Analysis Complete");
    } catch (e) {
      setStatus(`Error: ${e}`);
      console.error(e);
    } finally {
      setProcessingSide(null);
    }
  };

  const runRight = async () => {
    if (processingSide) return;
    setProcessingSide("right");
    try {
      const res = await analyze(inputText, rightModelId);
      setRightResult(res);
      setStatus("Analysis Complete");
    } catch (e) {
      setStatus(`Error: ${e}`);
      console.error(e);
    } finally {
      setProcessingSide(null);
    }
  };

  const runBoth = async () => {
    if (processingSide) return;
    setProcessingSide("both");
    try {
      // Run sequentially to manage VRAM/Engine state
      const resLeft = await analyze(inputText, leftModelId);
      setLeftResult(resLeft);

      await new Promise((r) => setTimeout(r, 100)); // UI Breath

      const resRight = await analyze(inputText, rightModelId);
      setRightResult(resRight);

      setStatus("Comparison Complete");
    } catch (e) {
      setStatus(`Error: ${e}`);
      console.error(e);
    } finally {
      setProcessingSide(null);
    }
  };

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1>WebLLM Model Comparison</h1>
        <p>Compare perplexity and logprobs across different models.</p>

        <div style={styles.statusBox}>
          <strong>Status:</strong> {status}
          {progress > 0 && progress < 100 && (
            <div style={styles.progressBar}>
              <div style={{ ...styles.progressFill, width: `${progress}%` }} />
            </div>
          )}
        </div>
      </header>

      <div style={styles.controls}>
        <div style={styles.controlGroup}>
          <label>Input Text:</label>
          <div style={styles.presets}>
            <button
              style={styles.tinyBtn}
              onClick={() => setInputText(EASY_TEXT)}
            >
              Populate Easy
            </button>
            <button
              style={styles.tinyBtn}
              onClick={() => setInputText(DIFFICULT_TEXT)}
            >
              Populate Difficult
            </button>
          </div>
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            style={styles.textarea}
            rows={4}
          />
        </div>

        <div style={styles.buttonGroup}>
          <button
            style={styles.mainBtn}
            onClick={runBoth}
            disabled={!!processingSide}
          >
            {processingSide === "both"
              ? "Running Model 2..."
              : "Run Both & Compare"}
          </button>
        </div>
      </div>

      <div style={styles.comparisonGrid}>
        {/* Left Panel */}
        <div style={styles.panel}>
          <div style={styles.panelHeader}>
            <select
              value={leftModelId}
              onChange={(e) => setLeftModelId(e.target.value)}
              style={styles.select}
              disabled={!!processingSide}
            >
              {AVAILABLE_MODELS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
            <button
              onClick={runLeft}
              disabled={!!processingSide}
              style={styles.runBtn}
            >
              Run
            </button>
          </div>
          <LogprobView
            title="Result A"
            result={leftResult}
            isLoading={processingSide === "left" || processingSide === "both"}
          />
        </div>

        {/* Right Panel */}
        <div style={styles.panel}>
          <div style={styles.panelHeader}>
            <select
              value={rightModelId}
              onChange={(e) => setRightModelId(e.target.value)}
              style={styles.select}
              disabled={!!processingSide}
            >
              {AVAILABLE_MODELS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
            <button
              onClick={runRight}
              disabled={!!processingSide}
              style={styles.runBtn}
            >
              Run
            </button>
          </div>
          <LogprobView
            title="Result B"
            result={rightResult}
            isLoading={processingSide === "right"}
          />
        </div>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: 20,
    maxWidth: 1400,
    margin: "0 auto",
    fontFamily: "system-ui, sans-serif",
    color: "#333",
  },
  header: {
    textAlign: "center",
    marginBottom: 30,
  },
  statusBox: {
    background: "#f0f0f0",
    padding: 10,
    borderRadius: 6,
    maxWidth: 600,
    margin: "15px auto",
    fontSize: 14,
  },
  progressBar: {
    height: 4,
    background: "#ddd",
    borderRadius: 2,
    marginTop: 8,
    overflow: "hidden",
  },
  progressFill: {
    height: "100%",
    background: "#2196F3",
    transition: "width 0.2s",
  },
  controls: {
    background: "#fff",
    padding: 20,
    borderRadius: 8,
    border: "1px solid #eee",
    marginBottom: 30,
    boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
  },
  controlGroup: {
    marginBottom: 15,
  },
  textarea: {
    width: "100%",
    padding: 10,
    borderRadius: 6,
    border: "1px solid #ddd",
    fontSize: 14,
    marginTop: 5,
    fontFamily: "inherit",
  },
  presets: {
    marginBottom: 5,
    display: "flex",
    gap: 10,
  },
  tinyBtn: {
    fontSize: 12,
    padding: "4px 8px",
    background: "#eee",
    border: "none",
    borderRadius: 4,
    cursor: "pointer",
  },
  buttonGroup: {
    textAlign: "center",
  },
  mainBtn: {
    padding: "12px 30px",
    fontSize: 16,
    background: "#2196F3",
    color: "white",
    border: "none",
    borderRadius: 6,
    cursor: "pointer",
    fontWeight: "bold",
  },
  comparisonGrid: {
    display: "flex",
    gap: 20,
  },
  panel: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    gap: 15,
  },
  panelHeader: {
    display: "flex",
    gap: 10,
  },
  select: {
    flex: 1,
    padding: 10,
    borderRadius: 6,
    border: "1px solid #ddd",
    fontSize: 14,
  },
  runBtn: {
    padding: "0 20px",
    background: "#666",
    color: "white",
    border: "none",
    borderRadius: 6,
    cursor: "pointer",
  },
  resultBoxEmpty: {
    padding: 40,
    background: "#f9f9f9",
    borderRadius: 8,
    border: "1px dashed #ccc",
    textAlign: "center",
    color: "#888",
    height: 300,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  resultContainer: {
    background: "#fff",
    borderRadius: 8,
    border: "1px solid #ddd",
    padding: 15,
    boxShadow: "0 2px 4px rgba(0,0,0,0.05)",
  },
  statsContainer: {
    display: "flex",
    justifyContent: "space-around",
    background: "#e3f2fd",
    padding: 15,
    borderRadius: 8,
    marginBottom: 15,
  },
  tokensContainer: {
    lineHeight: 2.2,
    maxHeight: 400,
    overflowY: "auto",
    display: "flex",
    flexWrap: "wrap",
    gap: 4,
    padding: 4,
  },
  token: {
    padding: "2px 5px",
    borderRadius: 4,
    fontSize: 14,
    border: "1px solid rgba(0,0,0,0.05)",
    cursor: "help",
  },
};

export default App;
