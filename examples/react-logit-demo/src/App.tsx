import { useState, useEffect, useCallback } from "react";
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
}

// Color scale from green (high prob) to red (low prob)
function getColorForLogprob(logprob: number): string {
  const normalized = Math.max(0, Math.min(1, (logprob + 10) / 10));
  const hue = normalized * 120;
  return "hsl(" + hue + ", 70%, 85%)";
}

// Use custom model with prefill_all_logits for efficient input logprobs
const CUSTOM_MODEL_ID = "TinyLlama-1.1B-Chat-v1.0-q4f16_1-prefill-all-logits";

function LogprobView({
  title,
  result,
}: {
  title: string;
  result: AnalysisResult | null;
}) {
  if (!result) {
    return (
      <div
        style={{
          flex: 1,
          padding: 20,
          background: "#f9f9f9",
          borderRadius: 8,
          border: "1px dashed #ccc",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#888",
        }}
      >
        Waiting for analysis...
      </div>
    );
  }

  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        gap: 20,
        padding: 15,
        background: "#fff",
        borderRadius: 8,
        border: "1px solid #eee",
        boxShadow: "0 2px 4px rgba(0,0,0,0.05)",
      }}
    >
      <h2 style={{ margin: 0, fontSize: 18, color: "#333" }}>{title}</h2>

      {result.stats && (
        <div
          style={{
            background: "#e3f2fd",
            padding: 15,
            borderRadius: 8,
            display: "flex",
            justifyContent: "space-around",
            alignItems: "center",
          }}
        >
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 12, color: "#666" }}>Perplexity</div>
            <div style={{ fontSize: 24, fontWeight: "bold", color: "#2196F3" }}>
              {result.stats.perplexity.toFixed(2)}
            </div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 12, color: "#666" }}>Avg Logprob</div>
            <div style={{ fontSize: 18, fontWeight: "bold", color: "#333" }}>
              {result.stats.avg.toFixed(2)}
            </div>
          </div>
        </div>
      )}

      <div
        style={{
          background: "#fff",
          padding: 15,
          borderRadius: 8,
          border: "1px solid #ddd",
          lineHeight: 2.2,
          display: "flex",
          flexWrap: "wrap",
          gap: 2,
          color: "#333",
          maxHeight: 400,
          overflowY: "auto",
        }}
      >
        {result.tokens.map((tl, idx) => (
          <span
            key={idx}
            title={
              "logprob: " +
              tl.logprob.toFixed(4) +
              "\nprob: " +
              (Math.exp(tl.logprob) * 100).toFixed(2) +
              "%"
            }
            style={{
              background: getColorForLogprob(tl.logprob),
              padding: "2px 4px",
              borderRadius: 3,
              fontSize: 14,
              cursor: "help",
              border: "1px solid rgba(0,0,0,0.1)",
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
  const [status, setStatus] = useState("Idle");
  const [progress, setProgress] = useState(0);
  const [engine, setEngine] = useState<webllm.MLCEngineInterface | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [usesEfficientPrefill, setUsesEfficientPrefill] = useState(false);

  const [easyResult, setEasyResult] = useState<AnalysisResult | null>(null);
  const [hardResult, setHardResult] = useState<AnalysisResult | null>(null);

  useEffect(() => {
    async function init() {
      setStatus("Loading TinyLlama model with prefill_all_logits...");
      try {
        const appConfig = {
          model_list: [
            {
              model:
                "https://huggingface.co/mlc-ai/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC",
              model_id: CUSTOM_MODEL_ID,
              model_lib: "/TinyLlama-1.1B.wasm",
              vram_required_MB: 700,
              low_resource_required: true,
            },
          ],
        };

        const eng = await webllm.CreateMLCEngine(CUSTOM_MODEL_ID, {
          initProgressCallback: (report) => {
            setStatus(report.text);
            setProgress(report.progress * 100);
            if (report.text.includes("prefill_all_logits")) {
              setUsesEfficientPrefill(true);
            }
          },
          appConfig: appConfig,
        });
        setEngine(eng);
        setStatus('Ready - Click "Run Comparison" to analyze both texts');
      } catch (e) {
        setStatus("Error: " + String(e));
        console.error(e);
      }
    }
    init();
  }, []);

  const analyzeText = async (text: string): Promise<AnalysisResult> => {
    if (!engine) throw new Error("Engine not ready");

    const messages: webllm.ChatCompletionMessageParam[] = [
      { role: "user", content: text },
    ];

    const reply = await engine.chat.completions.create({
      messages,
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
      } else {
        for (let i = 0; i < inputLogprobs.length; i++) {
          tokens.push({
            token: "[" + i + "]",
            logprob: inputLogprobs[i],
          });
        }
      }

      const logprobValues = inputLogprobs.filter(
        (lp: unknown) => typeof lp === "number" && !isNaN(lp as number),
      ) as number[];

      let stats = null;
      if (logprobValues.length > 0) {
        const avg =
          logprobValues.reduce((a, b) => a + b, 0) / logprobValues.length;
        stats = {
          min: Math.min(...logprobValues),
          max: Math.max(...logprobValues),
          avg: avg,
          perplexity: Math.exp(-avg),
        };
      }

      return { text, tokens, stats };
    }
    throw new Error("No logprobs returned");
  };

  const runComparison = useCallback(async () => {
    if (!engine) return;

    setIsProcessing(true);
    setEasyResult(null);
    setHardResult(null);

    try {
      setStatus("Analyzing Easy Text...");
      const easy = await analyzeText(EASY_TEXT);
      setEasyResult(easy);

      // Small delay to let UI update
      await new Promise((r) => setTimeout(r, 100));

      setStatus("Analyzing Difficult Text...");
      const hard = await analyzeText(DIFFICULT_TEXT);
      setHardResult(hard);

      setStatus("Comparison Complete!");
    } catch (e) {
      setStatus("Error: " + String(e));
      console.error(e);
    } finally {
      setIsProcessing(false);
    }
  }, [engine]);

  return (
    <div
      style={{
        padding: 20,
        maxWidth: 1400,
        margin: "0 auto",
        fontFamily: "system-ui",
        color: "#333",
        backgroundColor: "#fff",
        minHeight: "100vh",
      }}
    >
      <div style={{ textAlign: "center", marginBottom: 30 }}>
        <h1>WebLLM Logprob Discovery</h1>
        <p style={{ color: "#666" }}>
          Side-by-side comparison of model perplexity on easy vs. difficult
          text.
          {usesEfficientPrefill && (
            <span
              style={{
                marginLeft: 10,
                padding: "2px 8px",
                background: "#4CAF50",
                color: "white",
                borderRadius: 4,
                fontSize: 12,
              }}
            >
              ⚡ Using prefill_all_logits
            </span>
          )}
        </p>

        <div
          style={{
            background: "#f5f5f5",
            padding: 10,
            borderRadius: 8,
            marginBottom: 20,
            maxWidth: 600,
            margin: "0 auto 20px auto",
            color: "#333",
          }}
        >
          <p style={{ margin: 0, fontSize: 14 }}>
            <strong>Status:</strong> {status}
          </p>
          {progress > 0 && progress < 100 && (
            <div
              style={{
                marginTop: 8,
                background: "#ddd",
                borderRadius: 4,
                overflow: "hidden",
                height: 4,
              }}
            >
              <div
                style={{
                  width: progress + "%",
                  height: "100%",
                  background: "#4CAF50",
                  transition: "width 0.3s",
                }}
              />
            </div>
          )}
        </div>

        <button
          onClick={runComparison}
          disabled={!engine || isProcessing}
          style={{
            padding: "12px 32px",
            fontSize: 18,
            background: engine && !isProcessing ? "#2196F3" : "#ccc",
            color: "white",
            border: "none",
            borderRadius: 6,
            cursor: engine && !isProcessing ? "pointer" : "not-allowed",
            boxShadow: "0 2px 5px rgba(0,0,0,0.2)",
          }}
        >
          {isProcessing ? "Running Analysis..." : "Run Comparison"}
        </button>
      </div>

      <div
        style={{
          display: "flex",
          gap: 20,
          flexDirection: "row",
          alignItems: "flex-start",
        }}
      >
        <LogprobView title="Easy Text (Predictable)" result={easyResult} />
        <LogprobView title="Difficult Text (Abstract)" result={hardResult} />
      </div>

      <div
        style={{
          marginTop: 40,
          padding: 20,
          background: "#fff3e0",
          borderRadius: 8,
          color: "#333",
          fontSize: 14,
        }}
      >
        <h4 style={{ margin: "0 0 10px 0" }}>What to look for:</h4>
        <ul style={{ margin: 0, paddingLeft: 20, lineHeight: 1.6 }}>
          <li>
            <strong>Perplexity:</strong> The "Easy Text" should have a
            significantly lower perplexity score (closer to 1.0) than the
            "Difficult Text".
          </li>
          <li>
            <strong>Visual Density:</strong> The "Easy Text" should be mostly
            green (high probability tokens). The "Difficult Text" should have
            more orange/red tokens (unexpected words).
          </li>
          <li>
            <strong>Tokenization:</strong> Notice how complex words in the
            difficult text might be split into multiple tokens, each with
            potentially lower probability.
          </li>
        </ul>
      </div>
    </div>
  );
}

export default App;
