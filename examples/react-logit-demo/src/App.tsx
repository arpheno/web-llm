import { useState, useEffect, useCallback } from "react";
import * as webllm from "@mlc-ai/web-llm";

// 200-word sample prompt for annotation
const SAMPLE_PROMPT =
  "The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain. The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true. Eventually, it became obvious that commercial developers and researchers had grossly underestimated the difficulty of the project. In 1974, in response to the criticism from James Lighthill and ongoing pressure from congress, the U.S. and British Governments stopped funding undirected research into artificial intelligence, leading to what became known as an AI winter.";

interface TokenLogprob {
  token: string;
  logprob: number;
}

// Color scale from green (high prob) to red (low prob)
function getColorForLogprob(logprob: number): string {
  const normalized = Math.max(0, Math.min(1, (logprob + 10) / 10));
  const hue = normalized * 120;
  return "hsl(" + hue + ", 70%, 85%)";
}

// Use custom model with prefill_all_logits for efficient input logprobs
const CUSTOM_MODEL_ID = "TinyLlama-1.1B-Chat-v1.0-q4f16_1-prefill-all-logits";

function App() {
  const [status, setStatus] = useState("Idle");
  const [progress, setProgress] = useState(0);
  const [engine, setEngine] = useState<webllm.MLCEngineInterface | null>(null);
  const [tokenLogprobs, setTokenLogprobs] = useState<TokenLogprob[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [stats, setStats] = useState<{
    min: number;
    max: number;
    avg: number;
  } | null>(null);
  const [usesEfficientPrefill, setUsesEfficientPrefill] = useState(false);

  useEffect(() => {
    async function init() {
      setStatus("Loading TinyLlama model with prefill_all_logits...");
      try {
        // Custom app config using local WASM with prefill_all_logits support
        const appConfig = {
          model_list: [
            {
              // Model weights from HuggingFace (q4f16_1 version)
              model:
                "https://huggingface.co/mlc-ai/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC",
              model_id: CUSTOM_MODEL_ID,
              // Custom WASM with prefill_all_logits function - serve from public folder
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
            // Check if prefill_all_logits loaded
            if (report.text.includes("prefill_all_logits")) {
              setUsesEfficientPrefill(true);
            }
          },
          appConfig: appConfig,
        });
        setEngine(eng);
        setStatus('Ready - Click "Annotate Prompt" to analyze');
      } catch (e) {
        setStatus("Error: " + String(e));
        console.error(e);
      }
    }
    init();
  }, []);

  const annotatePrompt = useCallback(async () => {
    if (!engine) return;

    setIsProcessing(true);
    setStatus("Computing logprobs for input tokens...");
    setTokenLogprobs([]);
    setStats(null);

    try {
      const messages: webllm.ChatCompletionMessageParam[] = [
        { role: "user", content: SAMPLE_PROMPT },
      ];

      const reply = await engine.chat.completions.create({
        messages,
        max_tokens: 1,
        return_input_logprobs: true,
        logprobs: true,
        top_logprobs: 1,
      });

      const inputLogprobs = reply.input_logprobs;

      if (inputLogprobs && Array.isArray(inputLogprobs)) {
        const tokens: TokenLogprob[] = [];

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

        setTokenLogprobs(tokens);

        const logprobValues = inputLogprobs.filter(
          (lp: unknown) => typeof lp === "number" && !isNaN(lp as number),
        ) as number[];
        if (logprobValues.length > 0) {
          setStats({
            min: Math.min(...logprobValues),
            max: Math.max(...logprobValues),
            avg:
              logprobValues.reduce((a, b) => a + b, 0) / logprobValues.length,
          });
        }

        setStatus("Done! Analyzed " + inputLogprobs.length + " tokens");
      } else {
        setStatus(
          "No input logprobs returned - check if return_input_logprobs is supported",
        );
      }
    } catch (e) {
      setStatus("Error: " + String(e));
      console.error(e);
    } finally {
      setIsProcessing(false);
    }
  }, [engine]);

  const wordCount = SAMPLE_PROMPT.split(" ").length;

  return (
    <div
      style={{
        padding: 20,
        maxWidth: 1200,
        margin: "0 auto",
        fontFamily: "system-ui",
        color: "#333",
        backgroundColor: "#fff",
      }}
    >
      <h1>WebLLM Logprob Annotation Demo</h1>
      <p style={{ color: "#666" }}>
        Using <strong>{CUSTOM_MODEL_ID}</strong> to compute per-token logprobs
        for a {wordCount}-word prompt
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
            ⚡ Using prefill_all_logits (efficient)
          </span>
        )}
      </p>

      <div
        style={{
          background: "#f5f5f5",
          padding: 15,
          borderRadius: 8,
          marginBottom: 20,
          color: "#333",
        }}
      >
        <p style={{ margin: 0 }}>
          <strong>Status:</strong> {status}
        </p>
        {progress > 0 && progress < 100 && (
          <div
            style={{
              marginTop: 10,
              background: "#ddd",
              borderRadius: 4,
              overflow: "hidden",
            }}
          >
            <div
              style={{
                width: progress + "%",
                height: 8,
                background: "#4CAF50",
                transition: "width 0.3s",
              }}
            />
          </div>
        )}
      </div>

      <button
        onClick={annotatePrompt}
        disabled={!engine || isProcessing}
        style={{
          padding: "12px 24px",
          fontSize: 16,
          background: engine && !isProcessing ? "#2196F3" : "#ccc",
          color: "white",
          border: "none",
          borderRadius: 6,
          cursor: engine && !isProcessing ? "pointer" : "not-allowed",
          marginBottom: 20,
        }}
      >
        {isProcessing ? "Processing..." : "Annotate Prompt with Logprobs"}
      </button>

      <h3>Input Prompt ({wordCount} words):</h3>
      <div
        style={{
          background: "#fafafa",
          padding: 15,
          borderRadius: 8,
          border: "1px solid #ddd",
          lineHeight: 1.8,
          marginBottom: 20,
          color: "#333",
        }}
      >
        {SAMPLE_PROMPT}
      </div>

      {stats && (
        <div
          style={{
            background: "#e3f2fd",
            padding: 15,
            borderRadius: 8,
            marginBottom: 20,
            color: "#333",
          }}
        >
          <h4 style={{ margin: "0 0 10px 0" }}>Logprob Statistics:</h4>
          <p style={{ margin: 5 }}>
            <strong>Min:</strong> {stats.min.toFixed(4)} | <strong>Max:</strong>{" "}
            {stats.max.toFixed(4)} | <strong>Avg:</strong>{" "}
            {stats.avg.toFixed(4)}
          </p>
          <p style={{ margin: 5, fontSize: 12, color: "#666" }}>
            Logprobs closer to 0 = higher probability (green). More negative =
            lower probability (red).
          </p>
        </div>
      )}

      {tokenLogprobs.length > 0 && (
        <>
          <h3>Annotated Tokens ({tokenLogprobs.length} tokens):</h3>
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
            }}
          >
            {tokenLogprobs.map((tl, idx) => (
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

          <h3 style={{ marginTop: 30 }}>Detailed Token Table:</h3>
          <div
            style={{
              maxHeight: 400,
              overflow: "auto",
              border: "1px solid #ddd",
              borderRadius: 8,
              color: "#333",
            }}
          >
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                fontSize: 13,
              }}
            >
              <thead
                style={{ position: "sticky", top: 0, background: "#f5f5f5" }}
              >
                <tr>
                  <th
                    style={{
                      padding: 8,
                      textAlign: "left",
                      borderBottom: "2px solid #ddd",
                    }}
                  >
                    #
                  </th>
                  <th
                    style={{
                      padding: 8,
                      textAlign: "left",
                      borderBottom: "2px solid #ddd",
                    }}
                  >
                    Token
                  </th>
                  <th
                    style={{
                      padding: 8,
                      textAlign: "right",
                      borderBottom: "2px solid #ddd",
                    }}
                  >
                    Logprob
                  </th>
                  <th
                    style={{
                      padding: 8,
                      textAlign: "right",
                      borderBottom: "2px solid #ddd",
                    }}
                  >
                    Probability
                  </th>
                  <th
                    style={{
                      padding: 8,
                      textAlign: "left",
                      borderBottom: "2px solid #ddd",
                    }}
                  >
                    Visual
                  </th>
                </tr>
              </thead>
              <tbody>
                {tokenLogprobs.map((tl, idx) => (
                  <tr
                    key={idx}
                    style={{ background: idx % 2 === 0 ? "#fff" : "#fafafa" }}
                  >
                    <td style={{ padding: 8, borderBottom: "1px solid #eee" }}>
                      {idx}
                    </td>
                    <td
                      style={{
                        padding: 8,
                        borderBottom: "1px solid #eee",
                        fontFamily: "monospace",
                      }}
                    >
                      {JSON.stringify(tl.token)}
                    </td>
                    <td
                      style={{
                        padding: 8,
                        borderBottom: "1px solid #eee",
                        textAlign: "right",
                        fontFamily: "monospace",
                      }}
                    >
                      {tl.logprob.toFixed(6)}
                    </td>
                    <td
                      style={{
                        padding: 8,
                        borderBottom: "1px solid #eee",
                        textAlign: "right",
                        fontFamily: "monospace",
                      }}
                    >
                      {(Math.exp(tl.logprob) * 100).toFixed(4)}%
                    </td>
                    <td style={{ padding: 8, borderBottom: "1px solid #eee" }}>
                      <div
                        style={{
                          width: Math.max(5, Math.exp(tl.logprob) * 100) + "%",
                          height: 16,
                          background: getColorForLogprob(tl.logprob),
                          borderRadius: 3,
                          minWidth: 5,
                          maxWidth: "100%",
                        }}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      <div
        style={{
          marginTop: 40,
          padding: 20,
          background: "#fff3e0",
          borderRadius: 8,
          color: "#333",
        }}
      >
        <h4>How This Works:</h4>
        <ol style={{ lineHeight: 1.8 }}>
          <li>
            The {wordCount}-word prompt is sent to TinyLlama running in WebGPU
            via WebLLM
          </li>
          <li>
            Using a <strong>custom WASM</strong> with{" "}
            <code>prefill_all_logits</code> function, we get logits for ALL
            input tokens efficiently in one pass
          </li>
          <li>
            Using <code>return_input_logprobs: true</code>, we retrieve the
            log-probabilities for each input token
          </li>
          <li>
            Each token is colored based on its probability:{" "}
            <span
              style={{
                background: "hsl(120, 70%, 85%)",
                padding: "2px 6px",
                borderRadius: 3,
              }}
            >
              high prob
            </span>{" "}
            to{" "}
            <span
              style={{
                background: "hsl(0, 70%, 85%)",
                padding: "2px 6px",
                borderRadius: 3,
              }}
            >
              low prob
            </span>
          </li>
          <li>
            Hover over any token to see its exact logprob and probability
            percentage
          </li>
        </ol>
        <p style={{ fontSize: 12, color: "#666", marginTop: 10 }}>
          <strong>Use Case:</strong> This enables client-side perplexity
          scoring, uncertainty detection, and speed-reading apps without sending
          data to external APIs.
        </p>
        <p style={{ fontSize: 11, color: "#888", marginTop: 5 }}>
          <strong>Technical Note:</strong> Standard MLC models only return the
          last token's logit during prefill (optimized for generation). Our
          custom WASM includes <code>prefill_all_logits</code> which returns
          logits for all {wordCount > 0 ? wordCount + " " : ""}tokens in a
          single efficient GPU pass.
        </p>
      </div>
    </div>
  );
}

export default App;
