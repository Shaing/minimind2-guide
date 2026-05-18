# MiniMind2 推理內核互動學習指南

> 透過 470 行 PyTorch 原始碼，從第一性原理看懂 LLM 推理機制

**[🌐 GitHub Page](https://shaing.github.io/minimind2-guide/)**  
**[🌐 開啟互動學習指南](./index.html)** &nbsp;·&nbsp; **[English README](./README_en.md)**  

---

## 學習背景

### 為什麼選 MiniMind2？

市面上大多數 LLM 的推理邏輯藏在 Hugging Face Transformers 的多層抽象之下，初學者難以直接看清張量形狀如何隨著生成過程演化。

[MiniMind2](https://github.com/jingyaogong/minimind) 是一個從零開始、完全用 PyTorch 原生實作的極簡語言模型：

- 參數量：**25.8M**（GPT-3 的 1/7000）
- 核心模型只有 **~470 行**（[model_minimind.py](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py)），沒有任何第三方抽象層
- KV Cache 就是 3 行 `torch.cat`，Prefill/Decode 的切換只是一個 `if past_key_values is None`
- 所有概念都是**顯式可見**的，讓學習者直接對應程式碼行號理解原理

這份學習指南正是以此為基礎，把 LLM 推理最容易混淆的概念拆解成 9 個 Q&A，並附上即時互動的動畫與計算器。第 9 章 (Q9) 從半導體物理一路推導到 LLM 記憶體頻寬瓶頸，搭配 [llm-memory-bandwidth-report.md](./llm-memory-bandwidth-report.md) 作為深度補充。

---

## 學習指南章節

| 章節 | 主題 | 互動元件 |
| ------ | ------ | ---------- |
| Glossary | 核心張量符號速查（B、T、H、D_h…） | — |
| Q1 | **Prefill 預填充階段**：整段 prompt 一次並行處理 | 張量形狀追蹤 |
| Q2 | **Decode 解碼階段**：每次只生成一個 token | 逐步動畫 + KV Cache 成長視覺化 |
| Q3 | **KV Cache 原理**：為什麼能快取 K/V 但不能快取 Q？ | 記憶體用量即時計算器 |
| Q4 | **GQA 分組查詢注意力**：用更少的 KV heads 節省記憶體 | 頭配對視覺化 |
| Q5 | **多輪對話的本質**：LLM 是無狀態的，對話歷史靠 template 拼接 | 對話時間軸 Demo |
| Q6 | **Context Window 為何是 4096？** 三道物理閘門 + YaRN 延伸 | 長度 vs FLOPs/記憶體滑桿 |
| Q7 | **Causal Mask + Padding Mask**：兩種掩碼的組合方式 | 注意力掩碼網格 |
| Q8 | **串流輸出原理**：TTFT vs TPOT，TextStreamer 如何運作 | — |
| Q9 | **記憶體頻寬瓶頸**：從馮紐曼瓶頸 + Roofline 模型解釋為何 decode 是 memory-bound | Roofline 計算器 |
| Appendix | 一張圖串起所有概念 + 進階挑戰題 | — |

每個章節都直接標注對應的 [model_minimind.py](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py) 行號，方便對照閱讀原始碼。

---

## 使用方式

### 本地開啟

直接用瀏覽器開啟 `index.html`，無需任何伺服器或安裝。

```bash
# macOS
open index.html

# Linux
xdg-open index.html

# Windows
start index.html
```

### GitHub Pages

1. 將此資料夾推上 GitHub（建議 repo 名稱：`minimind2-guide`）
2. 進入 repo → **Settings → Pages**
3. Source 選 `Deploy from a branch`，Branch 選 `main`，資料夾選 `/ (root)`
4. 儲存後，學習指南公開網址為：

```text
https://{你的 GitHub 帳號}.github.io/minimind2-guide/index.html
```

---

## 相關資源

| 資源 | 連結 |
| ---- | ---- |
| MiniMind2 原始碼 | [jingyaogong/minimind](https://github.com/jingyaogong/minimind) |
| 核心模型實作 | [model/model_minimind.py](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py) |
| Chat Template | [MiniMind2/chat_template.jinja](https://github.com/jingyaogong/minimind/blob/master/MiniMind2/chat_template.jinja) |
| 模型設定 | [MiniMind2/config.json](https://github.com/jingyaogong/minimind/blob/master/MiniMind2/config.json) |
| 推理腳本 | [eval_llm.py](https://github.com/jingyaogong/minimind/blob/master/eval_llm.py) |
| Hugging Face 模型頁 | [jingyaogong/MiniMind2](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5) |

---

## 檔案結構

```text
minimind2-guide/
├── README.md                          # 本文件
├── README_en.md                       # English README
├── index.html                         # 互動式學習指南（繁中）
├── learning_guide_en.html             # 互動式學習指南（英文）— 全自帶 CSS/JS，無外部依賴
└── llm-memory-bandwidth-report.md     # Q9 深度補充：從半導體物理到 Roofline 模型
```
