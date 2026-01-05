# WebLLM Input Logprobs Feature

This project extends [WebLLM](https://github.com/mlc-ai/web-llm) to support **input token logprobs** - the ability to compute log-probabilities for every token in the input prompt, not just generated tokens.

## Why This Matters

Standard LLM inference is optimized for **generation**: during the "prefill" phase, only the logits for the *last* input token are computed (since that's all you need to predict the next token). However, for applications like:

- **Perplexity scoring** (measuring text difficulty/fluency)
- **Speed-reading annotation** (highlighting unpredictable words)
- **Uncertainty detection** (finding parts where the model is "surprised")
- **Client-side content analysis** (privacy-preserving text evaluation)

...you need logprobs for **all** input tokens.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         React Demo App                              │
│  (web-llm/examples/react-logit-demo)                                │
│                                                                     │
│  Uses: return_input_logprobs: true in chat.completions.create()     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        WebLLM TypeScript                            │
│  (web-llm/src/)                                                     │
│                                                                     │
│  Modified files:                                                    │
│  - src/llm_chat.ts: Calls prefill_all_logits when available         │
│  - src/engine.ts: Passes return_input_logprobs to pipeline          │
│  - src/config.ts: Adds return_input_logprobs option                 │
│  - src/openai_api_protocols/chat_completion.ts: Types for response  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Custom WASM Runtime                            │
│  (TinyLlama-1.1B.wasm with prefill_all_logits)                      │
│                                                                     │
│  Built from: mlc-llm/web/emcc/mlc_wasm_runtime_minimal.cc           │
│  Uses:       mlc-llm/python/mlc_llm/model/llama/llama_model.py      │
│              (defines prefill_all_logits TVM function)              │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Model-Level: `prefill_all_logits` Function

In `mlc-llm/python/mlc_llm/model/llama/llama_model.py`, we added a new function to the Llama model:

```python
def prefill_all_logits(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
    """Return logits for ALL input tokens, not just the last one."""
    # Standard forward pass through transformer layers
    hidden_states = self.model(input_embed, paged_kv_cache)
    # Apply LM head to ALL hidden states (not just [-1])
    logits = self.lm_head(hidden_states)
    return logits
```

This function is exported in the model's `exports` dict so it becomes available as a TVM PackedFunc in the compiled WASM.

### 2. WASM Runtime: Minimal Build

We created `mlc-llm/web/emcc/mlc_wasm_runtime_minimal.cc` - a lightweight runtime that includes only:

- TVM runtime core (`wasm_runtime.cc`)
- UTF-8 encoding support (`encoding.cc`)

This avoids linking the full MLC-LLM serve engine (which has C++ dependencies on tokenizers, xgrammar, etc. that don't work in browsers).

Built via:
```bash
cd mlc-llm/web
make minimal  # Produces dist/wasm/mlc_wasm_runtime_minimal.bc
```

### 3. Model Compilation

The model is compiled with `mlc_llm compile` using special flags:

```bash
mlc_llm compile ./mlc-chat-config.json \
  --device webgpu \
  -o TinyLlama-1.1B-minimal.wasm \
  --overrides "tensor_parallel_shards=1" \
  --lib-format wasm
```

The compilation uses the patched `tvm/contrib/emcc.py` to link against our minimal runtime instead of the full one.

### 4. WebLLM TypeScript Integration

In `web-llm/src/llm_chat.ts`, the key logic:

```typescript
// Check if model has prefill_all_logits function
const hasPrefillAllLogits = this.tvm.getFunction("prefill_all_logits") !== null;

if (options.return_input_logprobs && hasPrefillAllLogits) {
  // Call the efficient single-pass function
  const allLogits = this.prefillAllLogits(inputTokens);
  // Convert logits → logprobs via softmax
  const logprobs = this.computeLogprobsFromLogits(allLogits, inputTokens);
  response.input_logprobs = logprobs;
  response.input_tokens = inputTokens;
}
```

### 5. OpenAI-Compatible API

The feature is exposed via standard OpenAI chat completion API:

```typescript
const response = await engine.chat.completions.create({
  messages: [{ role: "user", content: "Your text here" }],
  max_tokens: 1,
  return_input_logprobs: true,  // ← NEW OPTION
});

// Response includes:
response.input_logprobs  // number[] - logprob for each input token
response.input_tokens    // string[] - the actual tokens
```

## Demo Application

Located in `web-llm/examples/react-logit-demo/`:

### Features
- **Side-by-side comparison**: Analyzes "easy" text vs "difficult" text
- **Visual annotation**: Tokens colored green (high prob) → red (low prob)
- **Perplexity scoring**: Computes perplexity = exp(-avg_logprob)

### Running the Demo

```bash
cd web-llm/examples/react-logit-demo
npm install
npm run dev
# Open http://localhost:5173
```

### Key Files

- `src/App.tsx`: Main React component with comparison UI
- `public/TinyLlama-1.1B.wasm`: The custom-compiled WASM with `prefill_all_logits`

## Files Changed

### mlc-llm Repository (committed)

| File | Purpose |
|------|---------|
| `web/Makefile` | Added `minimal` target for building lightweight runtime |
| `web/emcc/mlc_wasm_runtime_minimal.cc` | New minimal runtime (no serve engine) |
| `python/mlc_llm/model/llama/llama_model.py` | Added `prefill_all_logits` function |

### web-llm Repository (committed)

| File | Purpose |
|------|---------|
| `src/config.ts` | Added `return_input_logprobs` config option |
| `src/engine.ts` | Pass option through to LLMChat |
| `src/llm_chat.ts` | Core logic to call `prefill_all_logits` and compute logprobs |
| `src/openai_api_protocols/chat_completion.ts` | Type definitions for `input_logprobs` |
| `examples/react-logit-demo/` | Complete demo application |

### Uncommitted Changes (mlc-llm)

The following files have experimental changes for a C++ server-side implementation of input logprobs. These are **not needed** for the browser demo:

- `cpp/serve/config.cc` - Server-side config parsing
- `cpp/serve/engine_actions/new_request_prefill.cc` - Server-side logprob extraction
- `web/emcc/mlc_wasm_runtime.cc` - Full runtime with serve engine (bloated, not used)

These can be safely discarded with `git checkout -- <file>`.

## Performance

The `prefill_all_logits` approach is efficient because:

1. **Single GPU pass**: All logits computed in one forward pass
2. **No re-tokenization**: Uses the same tokens as normal prefill
3. **Minimal overhead**: Only difference from standard prefill is applying LM head to all positions instead of just the last

For a ~100 token input on TinyLlama-1.1B with WebGPU, expect ~50-100ms for the full logprob computation.

## Integrating Into Your Existing App

If you already have an app that uses `@mlc-ai/web-llm` as an npm dependency, here's exactly what you need to do to add input logprobs support.

### What You Need to Ship

You need **two things** beyond what standard WebLLM provides:

| Asset | What it is | Where to get it | Size |
|-------|------------|-----------------|------|
| **Custom WASM** | Model compiled with `prefill_all_logits` function | `models/TinyLlama-1.1B-Chat-v1.0-q4f16_1/TinyLlama-1.1B-minimal.wasm` | ~6MB |
| **Modified WebLLM** | TypeScript library with `return_input_logprobs` support | `web-llm/lib/` (built output) | ~200KB |

The model weights (`.bin` shards) are **unchanged** - you use the same weights from HuggingFace.

### Option A: Use Local WebLLM Build (Recommended)

This is the cleanest approach - you replace the npm package with your local build.

#### Step 1: Build the modified WebLLM

```bash
cd /path/to/logitwebllm/web-llm
npm install
npm run build
```

This produces built files in `web-llm/lib/`.

#### Step 2: Link to your app

**Option A1: npm link (for development)**
```bash
cd /path/to/logitwebllm/web-llm
npm link

cd /path/to/your-app
npm link @mlc-ai/web-llm
```

**Option A2: file dependency (for production)**

In your app's `package.json`:
```json
{
  "dependencies": {
    "@mlc-ai/web-llm": "file:../logitwebllm/web-llm"
  }
}
```

Then run `npm install`.

**Option A3: Copy the built files**

Copy `web-llm/lib/` into your project and import directly:
```typescript
import * as webllm from "./lib/index.js";
```

#### Step 3: Host the custom WASM

Copy the WASM file to your app's public/static folder:
```bash
cp /path/to/logitwebllm/models/TinyLlama-1.1B-Chat-v1.0-q4f16_1/TinyLlama-1.1B-minimal.wasm \
   /path/to/your-app/public/TinyLlama-1.1B.wasm
```

#### Step 4: Configure your app to use custom WASM

```typescript
import * as webllm from "@mlc-ai/web-llm";

// Define custom model config pointing to YOUR hosted WASM
const appConfig = {
  model_list: [
    {
      // Standard weights from HuggingFace (unchanged)
      model: "https://huggingface.co/mlc-ai/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC",
      model_id: "TinyLlama-1.1B-logprobs",
      // YOUR custom WASM with prefill_all_logits
      model_lib: "/TinyLlama-1.1B.wasm",  // ← Served from your public folder
      vram_required_MB: 700,
      low_resource_required: true,
    },
  ],
};

// Initialize engine with custom config
const engine = await webllm.CreateMLCEngine("TinyLlama-1.1B-logprobs", {
  appConfig: appConfig,
  initProgressCallback: (report) => console.log(report.text),
});
```

#### Step 5: Use the new API

```typescript
const response = await engine.chat.completions.create({
  messages: [{ role: "user", content: "Text to analyze" }],
  max_tokens: 1,
  return_input_logprobs: true,  // ← Enable input logprobs
  logprobs: true,
  top_logprobs: 1,
});

// Access the results
const inputLogprobs: number[] = response.input_logprobs;  // [-2.3, -0.5, -1.2, ...]
const inputTokens: string[] = response.input_tokens;      // ["The", " sun", " is", ...]

// Calculate perplexity
const avgLogprob = inputLogprobs.reduce((a, b) => a + b, 0) / inputLogprobs.length;
const perplexity = Math.exp(-avgLogprob);
```

### Option B: Patch npm Package (Quick Hack)

If you don't want to manage a local WebLLM build, you can patch `node_modules`:

```bash
# After npm install, copy over modified files
cp /path/to/logitwebllm/web-llm/lib/* node_modules/@mlc-ai/web-llm/lib/
```

**Warning**: This gets overwritten on `npm install`. Use `patch-package` for persistence:
```bash
npx patch-package @mlc-ai/web-llm
```

### Hosting Requirements

| File | Hosting Location | Notes |
|------|------------------|-------|
| `TinyLlama-1.1B.wasm` | Your server's `/public` or CDN | Must be same-origin or CORS-enabled |
| Model weights | HuggingFace (default) or your CDN | ~700MB, cached in IndexedDB |

The WASM file **must** be served with correct MIME type:
```
Content-Type: application/wasm
```

Most static hosts (Vercel, Netlify, etc.) handle this automatically.

### Complete Integration Example

Here's a minimal working example:

```typescript
// app.ts
import * as webllm from "@mlc-ai/web-llm";

async function analyzeText(text: string) {
  const appConfig = {
    model_list: [{
      model: "https://huggingface.co/mlc-ai/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC",
      model_id: "TinyLlama-logprobs",
      model_lib: "/TinyLlama-1.1B.wasm",
      vram_required_MB: 700,
    }],
  };

  const engine = await webllm.CreateMLCEngine("TinyLlama-logprobs", { appConfig });

  const response = await engine.chat.completions.create({
    messages: [{ role: "user", content: text }],
    max_tokens: 1,
    return_input_logprobs: true,
  });

  return {
    tokens: response.input_tokens,
    logprobs: response.input_logprobs,
    perplexity: Math.exp(-response.input_logprobs.reduce((a, b) => a + b, 0) / response.input_logprobs.length),
  };
}

// Usage
const result = await analyzeText("The quick brown fox jumps over the lazy dog.");
console.log("Perplexity:", result.perplexity);
```

### Checklist Before Deploying

- [ ] Built WebLLM with `npm run build` in `web-llm/`
- [ ] Linked/copied WebLLM to your app (Option A1, A2, or A3)
- [ ] Copied `TinyLlama-1.1B-minimal.wasm` to your public folder
- [ ] Updated `appConfig` to point to your hosted WASM
- [ ] Tested locally with `npm run dev`
- [ ] Verified WASM loads (check Network tab - should be ~6MB)
- [ ] Verified `response.input_logprobs` is populated (not undefined)

### Troubleshooting

**"return_input_logprobs is not a valid option"**
→ You're using the npm version of WebLLM, not your modified build. Check your import.

**"input_logprobs is undefined"**
→ The model doesn't have `prefill_all_logits`. Check that you're using the custom WASM.

**WASM fails to load / 404**
→ Check the path in `model_lib`. It should be relative to your app's public root.

**"LinkError: import object field ... is not a Function"**
→ You're using the wrong WASM (one compiled with full runtime). Use `TinyLlama-1.1B-minimal.wasm`.

---

## Keeping In Sync With Upstream

This project forks two repositories. Here's what you need and how to stay updated.

### Do I Need Both Repositories?

| Repository | Purpose | When You Need It |
|------------|---------|------------------|
| **mlc-llm** | Compiling models with `prefill_all_logits` | Only when compiling new models or updating model architecture |
| **web-llm** | TypeScript runtime with `return_input_logprobs` | Always - this is what your app imports |

**For most use cases:** You only need `web-llm` after initial setup. The WASM files don't change unless you:
- Want to use a different model (e.g., Llama-3, Mistral)
- Need to update TVM/MLC for bug fixes or performance improvements

### Repository Structure

```
logitwebllm/
├── mlc-llm/          # Fork of github.com/mlc-ai/mlc-llm
│   ├── web/          # WASM build system
│   └── python/       # Model definitions (prefill_all_logits)
│
├── web-llm/          # Fork of github.com/mlc-ai/web-llm  
│   ├── src/          # TypeScript source (our changes live here)
│   └── lib/          # Built output (ship this)
│
├── models/           # Compiled WASM files
│   └── TinyLlama-1.1B-Chat-v1.0-q4f16_1/
│       └── TinyLlama-1.1B-minimal.wasm
│
└── README.md         # This file
```

### Setting Up Upstream Remotes

First time only - add upstream remotes:

```bash
# For web-llm
cd web-llm
git remote add upstream https://github.com/mlc-ai/web-llm.git
git fetch upstream

# For mlc-llm
cd ../mlc-llm
git remote add upstream https://github.com/mlc-ai/mlc-llm.git
git fetch upstream
```

Verify remotes:
```bash
git remote -v
# Should show:
# origin    git@github.com:YOUR_USER/web-llm.git (your fork)
# upstream  https://github.com/mlc-ai/web-llm.git (official)
```

### Syncing web-llm (TypeScript Changes)

This is the one you'll update most often.

```bash
cd web-llm

# Fetch latest upstream
git fetch upstream

# Check what changed upstream
git log --oneline main..upstream/main

# Merge upstream into your branch
git checkout main
git merge upstream/main

# If conflicts, they'll likely be in:
# - src/llm_chat.ts (our main changes)
# - src/config.ts
# - src/openai_api_protocols/chat_completion.ts
```

**Resolving conflicts:**

Our changes are localized to a few areas. When merging:

1. **Keep our additions** - `return_input_logprobs`, `prefill_all_logits` logic
2. **Take upstream changes** - everything else

After resolving:
```bash
git add .
git commit -m "Merge upstream web-llm"
npm run build  # Rebuild
```

### Syncing mlc-llm (Model Compilation)

Only needed when:
- Upstream fixes bugs in model compilation
- You want to compile a newer model architecture
- TVM runtime changes affect WASM

```bash
cd mlc-llm

git fetch upstream
git merge upstream/main

# Our changes are in:
# - python/mlc_llm/model/llama/llama_model.py (prefill_all_logits function)
# - web/Makefile (minimal target)
# - web/emcc/mlc_wasm_runtime_minimal.cc (minimal runtime)
```

After merging, you may need to **recompile the model**:
```bash
cd web
make minimal  # Rebuild the minimal runtime

# Then recompile your model (if model code changed)
source ../.venv/bin/activate
mlc_llm compile ./path/to/mlc-chat-config.json \
  --device webgpu \
  -o TinyLlama-1.1B-minimal.wasm
```

### What Our Changes Actually Are

**web-llm (4 files modified):**

| File | Change | Lines |
|------|--------|-------|
| `src/config.ts` | Added `return_input_logprobs` to config types | ~5 lines |
| `src/engine.ts` | Pass option through to LLMChat | ~3 lines |
| `src/llm_chat.ts` | Core logic: call `prefill_all_logits`, compute softmax | ~50 lines |
| `src/openai_api_protocols/chat_completion.ts` | Type definitions for response | ~10 lines |

**mlc-llm (3 files modified/added):**

| File | Change | Lines |
|------|--------|-------|
| `python/mlc_llm/model/llama/llama_model.py` | Added `prefill_all_logits` method + export | ~20 lines |
| `web/Makefile` | Added `minimal` target | ~10 lines |
| `web/emcc/mlc_wasm_runtime_minimal.cc` | New file - minimal WASM runtime | ~40 lines |

**Total: ~140 lines of changes** across both repos.

### Automation Script

Create a script to sync both repos:

```bash
#!/bin/bash
# sync-upstream.sh

set -e

echo "=== Syncing web-llm ==="
cd web-llm
git fetch upstream
git merge upstream/main --no-edit || {
    echo "Conflicts in web-llm! Resolve manually."
    exit 1
}
npm run build

echo "=== Syncing mlc-llm ==="
cd ../mlc-llm
git fetch upstream
git merge upstream/main --no-edit || {
    echo "Conflicts in mlc-llm! Resolve manually."
    exit 1
}

echo "=== Done! ==="
echo "If model compilation changed, run: cd mlc-llm/web && make minimal"
```

### When Upstream Adds input_logprobs Support

If the official WebLLM adds `return_input_logprobs` support (likely eventually), you can:

1. **Check if compatible:** Test if their API matches ours
2. **Migrate:** Switch to upstream npm package
3. **Keep custom WASM:** You may still need custom WASM if their model compilation doesn't include `prefill_all_logits`

Until then, maintain your fork.

### Recommended Sync Frequency

| Repo | Frequency | Reason |
|------|-----------|--------|
| web-llm | Monthly or on new features | TypeScript changes, bug fixes |
| mlc-llm | Quarterly or when needed | Model architecture changes are rare |

---

## Future Work

- Support for other model architectures (Mistral, Qwen, etc.)
- Streaming logprobs during generation
- Batch processing for multiple texts
- Integration with official WebLLM release
