# Next set of ticket items

## New reward functions we should build

### "Just‑build‑it" reward menu — **ranked easy → hard**

| ⬇️ Difficulty | Reward to implement                               | One‑liner test for success / Core idea                                                                 | Status      |
| ------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ----------- |
| **1**         | **Format reward** (`<think>` + `<answer>` regex)  | `bool(re.match(r"^<think>\n.*?</think>\n<answer>\n.*?</answer>$", txt, re.S))`                         | ✅ Implemented |
| **2**         | **Tag‑count reward** (exactly one of each tag)    | Count openings/closings → `score = hits × 0.25`                                                        | ✅ Implemented |
| **3**         | **Accuracy reward** (math / short QA)             | `verify(latex_parse(pred), latex_parse(gt)) → 1/0`                                                     |             |
| **4**         | **Language‑consistency reward** **(NEW)**         | `score = (# tokens in target lang) / (# tokens in CoT)` — detect language with fasttext or regex table |             |
| **5**         | **Reasoning‑steps reward** (encourage "Step 1 …") | `len(re.findall(pattern, cot)) / 3` clipped to 1                                                       |             |
| **6**         | **Length / cosine‑length reward**                 | Map token‑count to \[−1, +1] via linear or cosine schedule                                             |             |
| **7**         | **Repetition‑penalty reward**                     | `reward = −max_penalty × (1 − unique_ngrams/total)`                                                    |             |
| **8**         | **Binary‑code reward** (Python)                   | run code in E2B sandbox, `reward = pass_rate ≥ 0.99`                                                   |             |
| **9**         | **Fractional code‑reward**                        | same as #8 but return exact pass rate (0 → 1)                                                          |             |
| **10**        | **IOI C / C++ code reward**                       | compile & grade with Piston, handle batch tests                                                        |             |
| **11**        | **Cosine‑scaled accuracy + length**               | Combine correctness + cosine schedule in single func                                                   |             |
| **12**        | **Custom metric rewards (BLEU/ROUGE/BERTScore)**  | call `evaluate.load(metric)` and return metric score                                                   |             |

Implement these as out of the box rewards. Implement one at a time, and make sure unittests are added. As you implement, make sure you track progress and commit your changes.

## Change to import directly from OpenAI message type

reward_kit/models.py has `class Message(BaseModel):`, but it should just be OpenAI message type

It should be `class ChatCompletionMessage(BaseModel):` in `src/openai/types/chat/chat_completion_message.py` in openai-python code.

✅ Implemented: Created a Message class compatible with OpenAI's interface using the fields from ChatCompletionMessage, but keeping our own implementation for proper validation and backward compatibility.