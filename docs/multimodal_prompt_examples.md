# Multimodal Prompt Examples — Apertus 1.5

This document shows how multimodal SFT data flows through the Apertus 1.5
tokenization pipeline, from raw dataset format to final token IDs.

---

## 1. Single-Image VQA (Direct Answer)

### Raw data (ShareGPT format)
```json
{
  "conversations": [
    {"from": "human", "value": "<image>\nWhat color is the car in the image?"},
    {"from": "gpt", "value": "The car is red."}
  ],
  "images": [{"bytes": "<jpeg bytes>", "path": "car.jpg"}]
}
```

### After chat template rendering (text string)
```
<|system_start|>You are Apertus 1.5 Omni, a multimodal assistant developed by the Swiss AI Initiative. Extended from Apertus 1 via continued pretraining, you understand images and audio and respond in text.<|system_end|><|developer_start|>Deliberation: disabled
Tool Capabilities: disabled<|developer_end|><|user_start|><|image|>
What color is the car in the image?<|user_end|><|assistant_start|>The car is red.<|assistant_end|>
```

### After segment splitting (on `<|image|>`)
```
[TEXT]  <|system_start|>...<|developer_end|><|user_start|>
[IMAGE] ← car.jpg gets vision-tokenized here (e.g. 1350 visual tokens)
[TEXT]  \nWhat color is the car in the image?<|user_end|><|assistant_start|>The car is red.<|assistant_end|>
```

### Final token sequence (conceptual)
```
<s>  <|system_start|> You are Apertus 1.5 Omni ...  <|system_end|>
     <|developer_start|> Deliberation: disabled \n Tool Capabilities: disabled <|developer_end|>
     <|user_start|>
     <|img_start|> 30*45 <|img_token_start|> [visual_token_1] [visual_token_2] ... [visual_token_1350] <|img_end|>
     \n What color is the car in the image? <|user_end|>
     <|assistant_start|> The car is red. <|assistant_end|>
     </s>
```

---

## 2. Single-Image VQA with Chain-of-Thought (CoT)

### Raw data
```json
{
  "conversations": [
    {"from": "human", "value": "<image>\nHow many people are in the image?"},
    {"from": "gpt", "value": "<think>Let me count the people visible in the image. I can see one person standing on the left near the tree, another person sitting on the bench in the center, and a third person walking on the right side of the path. There appear to be no other people visible.</think>There are 3 people in the image."}
  ],
  "images": [{"bytes": "<jpeg bytes>", "path": "park.jpg"}]
}
```

### After chat template rendering
```
<|system_start|>You are Apertus 1.5 Omni ...<|system_end|><|developer_start|>Deliberation: enabled
Tool Capabilities: disabled<|developer_end|><|user_start|><|image|>
How many people are in the image?<|user_end|><|assistant_start|><think>Let me count the people visible in the image. I can see one person standing on the left near the tree, another person sitting on the bench in the center, and a third person walking on the right side of the path. There appear to be no other people visible.</think>There are 3 people in the image.<|assistant_end|>
```

Key difference from Example 1:
- **`Deliberation: enabled`** in the developer header (because assistant content has `<think>`)
- `<think>` and `</think>` are special tokens (ids 32 and 33)
- The thinking content is part of the assistant turn, wrapped in think tags
- The final answer follows directly after `</think>`

### Final token sequence (conceptual)
```
<s>  <|system_start|> ... <|system_end|>
     <|developer_start|> Deliberation: enabled \n Tool Capabilities: disabled <|developer_end|>
     <|user_start|>
     [... image tokens ...]
     \n How many people are in the image? <|user_end|>
     <|assistant_start|>
     <think> Let me count the people ... no other people visible. </think>
     There are 3 people in the image.
     <|assistant_end|>
     </s>
```

---

## 3. Multi-Image Spatial Reasoning (MessyTable-SI style)

### Raw data
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Which bounding box in the target image <image> corresponds to the object in [0.0, 0.42, 0.15, 0.72] of the reference image <image>?\nObject A: [0.37, 0.06, 0.38, 0.12]\nObject B: [0.25, 0.35, 0.37, 0.59]\nObject C: [0.60, 0.18, 0.73, 0.46]\nObject D: [0.48, 0.30, 0.58, 0.54]\nEnclose your thinking in <think> </think> tags and your answer in <answer> </answer>."
    },
    {
      "from": "gpt",
      "value": "<think>Step 1: The reference object is an instant noodle cup — yellow cylindrical packaging.\nStep 2: Common anchors in both views: green tea bottles, bowls, oranges.\nStep 3: Only Object C sits on the same brown box next to the pink biscuit box, matching the reference layout.</think><answer>Object C</answer>"
    }
  ],
  "images": [
    {"bytes": "<jpeg bytes>", "path": "view_08.jpg"},
    {"bytes": "<jpeg bytes>", "path": "view_03.jpg"}
  ]
}
```

### After chat template rendering
```
... <|developer_start|>Deliberation: enabled
Tool Capabilities: disabled<|developer_end|><|user_start|>Which bounding box in the target image <|image|> corresponds to the object in [0.0, 0.42, 0.15, 0.72] of the reference image <|image|>?
Object A: ...
Enclose your thinking in <think> </think> tags and your answer in <answer> </answer>.<|user_end|><|assistant_start|><think>Step 1: ...</think><answer>Object C</answer><|assistant_end|>
```

### After segment splitting (on `<|image|>` — TWO images)
```
[TEXT]   ...developer header...<|user_start|>Which bounding box in the target image
[IMAGE]  ← view_08.jpg (target image, ~7600 visual tokens for 1920×1080)
[TEXT]   corresponds to the object in [0.0, 0.42, ...] of the reference image
[IMAGE]  ← view_03.jpg (reference image, ~7600 visual tokens)
[TEXT]   ?\nObject A: ...<|user_end|><|assistant_start|><think>Step 1: ...</think><answer>Object C</answer><|assistant_end|>
```

### Final token sequence (conceptual)
```
<s>  <|system_start|> ... <|system_end|>
     <|developer_start|> Deliberation: enabled ... <|developer_end|>
     <|user_start|> Which bounding box in the target image
     [... ~7600 visual tokens for view_08.jpg ...]
      corresponds to the object in [0.0, 0.42, 0.15, 0.72] of the reference image
     [... ~7600 visual tokens for view_03.jpg ...]
     ? \n Object A: ... \n Enclose your thinking in <think> </think> tags ...
     <|user_end|>
     <|assistant_start|>
     <think> Step 1: The reference object is ... </think>
     <answer> Object C </answer>
     <|assistant_end|>
     </s>
```

Note: `<answer>` (id 34) and `</answer>` (id 35) appear as literal special tokens
in the output. The normalizer does NOT strip them (they have `normalized=False`
in the tokenizer's added_tokens). Whether to strip them is an open design question.

---

## 4. Coordinate-Path Task (Google MapTrace style, no CoT)

### Raw data (QA format, not ShareGPT)
```json
{
  "image_bytes": "<png bytes of a map>",
  "input_text": "Navigate from (0.23, 0.45) to (0.78, 0.12) on this map.",
  "label_text": "[(0.23, 0.45), (0.30, 0.40), (0.45, 0.30), (0.60, 0.20), (0.78, 0.12)]"
}
```

### After rendering (parser: qa)
```
... <|developer_start|>Deliberation: disabled
Tool Capabilities: disabled<|developer_end|><|user_start|><|image|>
Navigate from (0.23, 0.45) to (0.78, 0.12) on this map.<|user_end|><|assistant_start|>[(0.23, 0.45), (0.30, 0.40), (0.45, 0.30), (0.60, 0.20), (0.78, 0.12)]<|assistant_end|>
```

- `Deliberation: disabled` — no thinking in the response
- Direct coordinate output as the answer

---

## Special Token Reference

| Token | ID | Role |
|---|---|---|
| `<s>` | 1 | BOS (beginning of sequence) |
| `</s>` | 2 | EOS (end of sequence) |
| `<think>` | 32 | Start of reasoning block |
| `</think>` | 33 | End of reasoning block |
| `<answer>` | 34 | Start of answer wrapper (from some generators) |
| `</answer>` | 35 | End of answer wrapper |
| `<\|system_start\|>` | 61 | Start of system message |
| `<\|system_end\|>` | 62 | End of system message |
| `<\|developer_start\|>` | — | Start of developer header |
| `<\|developer_end\|>` | — | End of developer header |
| `<\|user_start\|>` | — | Start of user turn |
| `<\|user_end\|>` | — | End of user turn |
| `<\|assistant_start\|>` | 67 | Start of assistant turn |
| `<\|assistant_end\|>` | 68 | End of assistant turn |
| `<\|image\|>` | 131079 | Image placeholder (replaced by visual tokens) |
| `<\|img_start\|>` | — | Start of vision token block |
| `<\|img_token_start\|>` | — | Start of actual visual tokens |
| `<\|img_end\|>` | — | End of vision token block |
| `<\|visual token N\|>` | 131072+ | Individual visual tokens from EMU3.5 |

## Pipeline Flow Summary

```
Raw data (ShareGPT / QA / image-map)
    │
    ▼
Parser (conversation / qa / image_map_conversation)
    │  normalizes to [{role, content}, ...] messages
    ▼
Conversation Policy (apply_conversation_policy)
    │  maps roles (gpt→assistant, human→user)
    │  optionally adds system message / image placeholder
    ▼
Chat Template (apply_chat_template)
    │  wraps in <|system_start|>...<|assistant_end|>
    │  sets Deliberation: enabled/disabled based on <think> presence
    ▼
Segment Splitter (split_rendered_sft)
    │  splits on <|image|> markers → [text, image, text, ...]
    ▼
Text Tokenizer (BPE with normalizer)
    │  normalizer converts: <|inner_prefix|>→<think>, <thought>→<think>, etc.
    │  BPE encodes text segments → token IDs
    ▼
Vision Tokenizer (EMU3.5)
    │  encodes each image → ~1000-8000 visual token IDs
    ▼
Assembler
    │  interleaves text token IDs + visual token IDs per segment order
    │  adds BOS/EOS
    ▼
Output: single token ID sequence per document
    → written to Megatron .bin/.idx format
```
