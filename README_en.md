# MiniMind2 Inference Internals · Interactive Learning Guide

> Understand how LLM inference actually works — through 470 lines of pure-PyTorch source code

**[🌐 Open the interactive guide](./learning_guide_en.html)** &nbsp;·&nbsp; **[繁體中文版 README](./README.md)**

---

## Why this guide exists

Most LLM inference behavior is hidden under several layers of Hugging Face Transformers abstraction. For someone trying to learn how a model actually generates text, that abstraction is a wall — you can't watch the tensor shapes evolve, you can't point at a single line and say *this is the KV cache*.

[MiniMind2](https://github.com/jingyaogong/minimind) tears that wall down. It's a from-scratch, pure-PyTorch language model:

- **25.8M parameters** — roughly 1/7000 the size of GPT-3
- The complete model lives in a **single ~470-line file** ([model_minimind.py](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py)) with no third-party abstraction layer
- The KV cache is three `torch.cat` calls. The prefill-vs-decode switch is one `if past_key_values is None`. Multi-turn chat is a list of dicts passed to a Jinja template.
- Every "magical" concept is **plainly visible** — you can follow the math by line number

This guide treats those 470 lines as the source of truth. Eight Q&As walk you through the most commonly confused mechanics, each backed by working code and live interactive widgets in the browser.

---

## What's in the guide

| # | Topic | Interactive component |
| --- | ------ | ------------------------ |
| Glossary | Tensor shape reference (B, T, H, D_h…) | — |
| Q1 | **Prefill** — process the whole prompt in one shot | Tensor shape walkthrough |
| Q2 | **Decode** — generate one token at a time | Step-by-step animation + KV cache growth visualizer |
| Q3 | **KV Cache** — why we can cache K/V but not Q | Live memory-usage calculator |
| Q4 | **GQA** — Grouped-Query Attention, the memory shortcut | MHA vs GQA head-pairing visualizer |
| Q5 | **Multi-turn Chat** — why the LLM is stateless | Conversation timeline demo |
| Q6 | **Why is the context window 4096?** Three independent gates + YaRN extension | Length vs FLOPs / memory slider |
| Q7 | **Causal Mask + Padding Mask** — two masks, composable | Attention mask grid |
| Q8 | **Streaming Output** — TTFT vs TPOT, TextStreamer internals | — |
| Appendix | One diagram tying all 8 concepts together + next-step challenges | — |

Every section anchors back to specific line numbers in [model_minimind.py](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py) so you can read the prose and the code side by side.

---

## How to use it

### Open locally

The HTML is fully self-contained — no server, no install, no dependencies.

```bash
# macOS
open learning_guide_en.html

# Linux
xdg-open learning_guide_en.html

# Windows
start learning_guide_en.html
```

### Host on GitHub Pages

1. Push this folder to a GitHub repository (suggested name: `minimind2-guide`)
2. Open the repo → **Settings → Pages**
3. Under *Source*, pick `Deploy from a branch`; choose branch `main` and folder `/ (root)`
4. Save. The guide is then live at:

```text
https://{your-github-username}.github.io/minimind2-guide/learning_guide_en.html
```

The Traditional Chinese version is at `index.html`. A language switcher in the top-right of each page jumps between them.

---

## Related resources

| Resource | Link |
| -------- | ---- |
| MiniMind2 source | [jingyaogong/minimind](https://github.com/jingyaogong/minimind) |
| Core model implementation | [model/model_minimind.py](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py) |
| Chat template | [MiniMind2/chat_template.jinja](https://github.com/jingyaogong/minimind/blob/master/MiniMind2/chat_template.jinja) |
| Model config | [MiniMind2/config.json](https://github.com/jingyaogong/minimind/blob/master/MiniMind2/config.json) |
| Inference script | [eval_llm.py](https://github.com/jingyaogong/minimind/blob/master/eval_llm.py) |
| Hugging Face collection | [jingyaogong/MiniMind2](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5) |

---

## Repository layout

```text
minimind2-guide/
├── README.md               # Traditional Chinese README
├── README_en.md            # this file
├── index.html     # interactive guide (Traditional Chinese)
└── learning_guide_en.html  # interactive guide (English) — fully self-contained, no external assets
```
