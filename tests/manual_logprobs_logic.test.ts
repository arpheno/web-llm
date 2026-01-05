import { describe, expect, test } from "@jest/globals";

// Mock ImageURL type
type ImageURL = { url: string };
const IMAGE_EMBED_SIZE = 5; // Arbitrary for test

// The logic to test
function calculateInputLogprobs(
  logitsOnCPUArray: Float32Array,
  chunk: Array<Array<number> | ImageURL>,
  vocabSize: number,
  inputLogprobs: number[],
) {
  let currentOffset = 0;
  for (const item of chunk) {
    if (Array.isArray(item)) {
      const tokens = item;
      for (const token of tokens) {
        if (currentOffset > 0) {
          const prevLogitIndex = currentOffset - 1;
          const tokenLogits = logitsOnCPUArray.subarray(
            prevLogitIndex * vocabSize,
            (prevLogitIndex + 1) * vocabSize,
          );

          let maxVal = -Infinity;
          for (let j = 0; j < vocabSize; j++) {
            if (tokenLogits[j] > maxVal) maxVal = tokenLogits[j];
          }
          let sumExp = 0;
          for (let j = 0; j < vocabSize; j++) {
            sumExp += Math.exp(tokenLogits[j] - maxVal);
          }
          const logSumExp = Math.log(sumExp) + maxVal;
          const logProb = tokenLogits[token] - logSumExp;
          inputLogprobs.push(logProb);
        } else {
          // First token of the sequence, no logprob available from previous context in this chunk
          inputLogprobs.push(0.0);
        }
        currentOffset++;
      }
    } else {
      // ImageURL
      currentOffset += IMAGE_EMBED_SIZE;
    }
  }
}

describe("Input Logprobs Logic", () => {
  test("Simple text sequence", () => {
    const vocabSize = 3;
    // Tokens: [0, 1, 2]
    // Logits for token 0 (predicting token 1): [10, 20, 10] -> softmax -> [0, 1, 0] approx
    // Logits for token 1 (predicting token 2): [10, 10, 20] -> softmax -> [0, 0, 1] approx

    // logits array size: 2 * vocabSize (we have 3 tokens, so 2 predictions: 0->1, 1->2)
    // Wait, the logits array corresponds to the output of the model for the sequence.
    // If input is [A, B, C], model outputs logits for [A->?, B->?, C->?]
    // So we have 3 sets of logits.
    // logits[0] is prediction after seeing A. Should match B.
    // logits[1] is prediction after seeing B. Should match C.
    // logits[2] is prediction after seeing C. (Next token).

    // In the code:
    // currentOffset goes 0, 1, 2.
    // When processing token A (offset 0): push 0.0.
    // When processing token B (offset 1): use logits at offset 0 (prediction from A).
    // When processing token C (offset 2): use logits at offset 1 (prediction from B).

    const logits = new Float32Array([
      // Logits from token 0 (predicting token 1)
      10, 20, 10,
      // Logits from token 1 (predicting token 2)
      10, 10, 20,
      // Logits from token 2 (predicting next) - unused for input logprobs of this chunk
      5,
      5, 5,
    ]);

    const chunk = [[0, 1, 2]];
    const inputLogprobs: number[] = [];

    calculateInputLogprobs(logits, chunk, vocabSize, inputLogprobs);

    expect(inputLogprobs.length).toBe(3);
    expect(inputLogprobs[0]).toBe(0.0); // First token

    // Check logprob for token 1 given token 0
    // Logits: 10, 20, 10. Max: 20.
    // Exp: e^-10, e^0, e^-10. Sum: 1 + 2e^-10. LogSum: ~0.
    // LogProb: 20 - (0 + 20) = 0.
    // Actually: log(e^10 + e^20 + e^10) = 20 + log(e^-10 + 1 + e^-10) ~= 20.
    // Prob(1) = e^20 / sum. LogProb = 20 - 20 = 0.
    expect(inputLogprobs[1]).toBeCloseTo(0, 1);

    // Check logprob for token 2 given token 1
    // Logits: 10, 10, 20. Max: 20.
    // LogProb(2) ~= 0.
    expect(inputLogprobs[2]).toBeCloseTo(0, 1);
  });

  test("Text with Image", () => {
    const vocabSize = 3;
    // Chunk: [Text(0), Image, Text(1)]
    // Text(0): offset 0. Logprob 0.0.
    // Image: offset 0 -> 0 + 5 = 5.
    // Text(1): offset 5. Use logits at 4.

    // We need logits up to index 4.
    // Logits size: 6 * vocabSize.
    // 0: from Text(0)
    // 1..4: from Image (4 vectors)
    // 5: from Text(1)

    const logits = new Float32Array(6 * vocabSize).fill(0);

    // Set logits at index 4 (last part of image) to predict Text(1) which is token 1.
    // We want token 1 to have high prob.
    logits[4 * vocabSize + 1] = 100;

    const chunk: Array<Array<number> | ImageURL> = [
      [0],
      { url: "http://fake.com" },
      [1],
    ];

    const inputLogprobs: number[] = [];
    calculateInputLogprobs(logits, chunk, vocabSize, inputLogprobs);

    expect(inputLogprobs.length).toBe(2); // Only text tokens get logprobs pushed?
    // Wait, let's check the code.
    // for (const item of chunk) { if (Array.isArray(item)) { ... inputLogprobs.push ... } }
    // Yes, only text tokens are pushed.

    expect(inputLogprobs[0]).toBe(0.0); // Text(0)

    // Text(1) is at offset 5 (1 + 5 = 6th position in sequence? No.)
    // currentOffset starts at 0.
    // Item 0 (Text [0]):
    //   token 0. offset 0. push 0.0. offset becomes 1.
    // Item 1 (Image):
    //   offset += 5. offset becomes 6.
    // Item 2 (Text [1]):
    //   token 1. offset 6.
    //   prevLogitIndex = 5.
    //   Uses logits at index 5.

    // Wait, my manual trace above said "Use logits at 4".
    // If image is 5 embeddings.
    // Sequence: T0, I0, I1, I2, I3, I4, T1
    // Indices:  0,  1,  2,  3,  4,  5,  6
    // T0 predicts I0 (logit 0)
    // I0 predicts I1 (logit 1)
    // ...
    // I4 predicts T1 (logit 5)

    // Code trace:
    // T0: offset 0. push 0.0. offset++ -> 1.
    // Image: offset += 5 -> 6.
    // T1: offset 6. prevLogitIndex = 5.
    // So it uses logits at index 5.

    // So I should set logits at index 5.
    logits[5 * vocabSize + 1] = 100; // Predict token 1
    logits[4 * vocabSize + 1] = 0; // Reset the wrong one I set earlier

    // Re-run
    const inputLogprobs2: number[] = [];
    calculateInputLogprobs(logits, chunk, vocabSize, inputLogprobs2);

    expect(inputLogprobs2[1]).toBeCloseTo(0, 1);
  });
});
