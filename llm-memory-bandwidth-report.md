# 大語言模型推論的記憶體頻寬瓶頸：從第一性原理出發

> 一份從半導體物理推導到系統優化的技術 report
> 適合：對 LLM 推論系統感興趣的碩士級讀者

---

## 摘要

本報告從半導體製程的物理限制出發，逐層推導為何大語言模型 (LLM) 的 decode 階段會被記憶體頻寬 (memory bandwidth) 主導，而 prefill 階段卻能逼近計算極限。文中以 **算術強度 (arithmetic intensity)** 與 **Roofline 模型** 為核心分析工具，量化 GPU 與邊緣 NPU 兩種架構的瓶頸位置差異，並從第一性原理推導出當前主流優化技術 (batching、量化、GQA、speculative decoding) 為何有效。最終以 Llama-7B 為實例完成端到端的頻寬計算，並對應到 PIM、wafer-scale 等新興硬體方向。

---

## 1. 問題設定

> **核心觀察**：在現代加速器上，LLM decode 速度幾乎完全由 HBM/LPDDR 的頻寬決定，而非 FLOPs。

舉個經驗事實：H100 提供 ~989 TFLOPS (FP16 tensor) 與 ~3.35 TB/s 的 HBM3 頻寬。跑 Llama-7B decode 時，實測 throughput 約 100~150 tokens/s。若硬體完全用於計算，理論值應該高出兩個數量級。**為什麼算力大量閒置？**

要回答這個問題，必須先承認一個物理事實：**儲存與計算發生在不同的電路上**。

---

## 2. 第一性原理基礎

### 2.1 馮紐曼瓶頸：儲存與計算的物理分離

任何數位電路都必須回答兩個問題：
1. 0 與 1 存在哪裡？
2. 算術運算在哪裡執行？

由於電晶體的功能特性，這兩個任務需要**不同的電路設計**：

| 功能 | 主要結構 | 特性 |
|---|---|---|
| 儲存（DRAM cell） | 1 電容 + 1 電晶體 | 高密度、慢、需 refresh |
| 儲存（SRAM cell） | 6 電晶體 | 低密度（面積大 ~100×）、快、靜態保持 |
| 計算（MAC unit） | 加法器 + 乘法器 + 暫存器 | 邏輯密集、需要靠近暫存器 |

DRAM 製程與邏輯 (logic) 製程**互不相容**：DRAM 為了壓低成本，使用較大的電容尺寸與特殊金屬層；邏輯製程則追求最小通道長度與最快開關速度。因此 HBM 必須作為一顆「獨立晶片」堆疊在 GPU die 旁邊，透過 interposer 連接。

```
┌──────────────────────────────┐
│  GPU Package                 │
│                              │
│  ┌──────────┐   ┌─────────┐  │
│  │ GPU Die  │←→ │  HBM    │  │  ← 兩個獨立晶片
│  │ (logic + │   │ (DRAM   │  │     必須跨界搬資料
│  │  SRAM)   │   │  only)  │  │
│  └──────────┘   └─────────┘  │
└──────────────────────────────┘
```

這就是 **馮紐曼瓶頸 (von Neumann bottleneck)** 的物理根源：資料必須在兩個物理分離的電路間移動才能完成運算。任何當代主流加速器——GPU、TPU、NPU——都無法繞過這個事實，差別只在於「移動成本如何分攤」。

### 2.2 記憶體階層的物理與經濟取捨

由於 SRAM 每 bit 面積是 DRAM 的 ~100 倍，將完整模型放進片上 SRAM 在經濟上不可行。以 H100 為例：

| 層級 | 容量 | 頻寬 | 距離 MAC |
|---|---|---|---|
| 暫存器 (Register File) | 256 KB / SM | ~固定週期存取 | 0 |
| L1 / Shared SRAM | 256 KB / SM | ~33 TB/s | 同 SM |
| L2 SRAM | ~50 MB | ~5 TB/s | 同 die |
| **HBM3** | **80 GB** | **3.35 TB/s** | **off-chip** |

容量每往下一層增加 ~10³ 倍，頻寬就掉 ~10× 並付出更高的存取延遲。這是經濟與物理共同決定的取捨，**不是設計缺陷**。

### 2.3 算術強度與 Roofline 模型

定義 **算術強度 (Arithmetic Intensity, AI)**：

$$
I = \frac{\text{FLOPs 執行數}}{\text{Bytes 從 DRAM 讀取數}}
$$

對某加速器，定義 **轉折算術強度 (ridge point)**：

$$
I^* = \frac{\text{Peak FLOPs}}{\text{Peak Bandwidth}}
$$

H100 的 $I^* = 989 \times 10^{12} / 3.35 \times 10^{12} \approx 295$ FLOPs/byte。

**Roofline 模型** 給出實際 throughput 的上界：

$$
\text{Throughput} = \min(\text{Peak FLOPs}, \, I \times \text{Bandwidth})
$$

- 當 $I < I^*$：**memory-bound**，throughput 受頻寬限制
- 當 $I > I^*$：**compute-bound**，throughput 受 FLOPs 限制

這個 ~300 的門檻值是後續所有分析的標尺。

---

## 3. Transformer 推論的計算結構

### 3.1 LLM 權重組成

一個標準的 decoder-only Transformer (以 Llama-7B 為例) 的權重組成：

```
參數 (FP16, 2 bytes each)
├── Token Embedding         : 32000 × 4096           = 131 M
├── Transformer Block × 32
│   ├── RMSNorm pre-attn   : 4096                    (negligible)
│   ├── W_Q, W_K, W_V, W_O : 4 × 4096²              = 67 M
│   ├── RMSNorm pre-ffn    : 4096                    (negligible)
│   └── SwiGLU FFN         : 3 × 4096 × 11008       = 135 M
├── Final RMSNorm          : 4096
└── LM Head                : 4096 × 32000           = 131 M (常與 embedding 共享)

總計 ≈ 6.7 B 參數 × 2 bytes ≈ 14 GB
```

**關鍵觀察**：FFN 佔每層 ~67% 的參數量。`搬權重` 的主體不是 attention，而是 FFN 的三個大矩陣。

### 3.2 KV Cache：語言模型獨有的狀態

KV cache 是推論時為了避免重算前文 attention 而保存的中間狀態。其大小：

$$
\text{KV} = 2 \cdot L \cdot S \cdot d_{\text{model}} \cdot b
$$

其中 $L$ 為層數，$S$ 為 sequence 長度，$d_{\text{model}}$ 為隱藏維度，$b$ 為每元素 byte 數。

Llama-7B (FP16) 的 KV cache 量級：

| Context 長度 | KV Cache 大小 |
|---|---|
| 2 K  tokens | ~1 GB |
| 8 K  tokens | ~4 GB |
| 32 K tokens | ~16 GB |
| 128 K tokens | ~64 GB |

**KV cache 不是權重**——它是 activation/state，會隨 context 線性增長。長 context 時，KV cache 可能比權重本身還大。

### 3.3 Prefill 與 Decode 的本質差異

對單一線性層 $Y = XW$，$W \in \mathbb{R}^{d \times d}$：

**Prefill**：輸入 $X \in \mathbb{R}^{N \times d}$ (N 個 prompt tokens)
- 讀取：$d^2 \cdot b$ bytes (整個 W) + $Nd \cdot b$ bytes (X)
- 計算：$2Nd^2$ FLOPs
- 算術強度：$I \approx \dfrac{2Nd^2}{d^2 \cdot b} = \dfrac{2N}{b}$

**Decode**：輸入 $x \in \mathbb{R}^{1 \times d}$ (1 個 token)
- 讀取：$d^2 \cdot b$ bytes (整個 W)
- 計算：$2d^2$ FLOPs
- 算術強度：$I \approx \dfrac{2d^2}{d^2 \cdot b} = \dfrac{2}{b}$

代入 FP16 (b=2)：

| 階段 | 算術強度 | 對比 $I^* \approx 295$ |
|---|---|---|
| Prefill (N=2000) | ~2000 | $\gg I^*$ → compute-bound |
| Decode (N=1) | ~1 | $\ll I^*$ → memory-bound |

**這就是 prefill 與 decode 的根本分歧**：兩者執行的數學運算其實是同一個 GEMM，只是 input 的 batch 維度差了三個數量級，導致算術強度也差了三個數量級。

### 3.4 權重共用的內在結構：為何 GEMM 能攤平搬運成本

前述分析顯示 prefill 與 decode 的算術強度相差三個數量級，差異來源於 input batch 維度。但這個觀察容易引出一個誤解：以為「共用是由 batching 創造出來的」。實際方向恰好相反——**權重的共用潛力是模型結構天生賦予的，batching 只是兌現它的硬體手段**。

對某層線性變換 $Y = XW$，無論輸入有多少 token，每個 token $i$ 的輸出 $y_i = x_i W$ 所使用的 $W$ 數值完全相同。因此 N 個 token 共用同一個 $W$ 這件事，**在任何 schedule 下都成立**——差別僅在硬體是否將這個共用實現出來。對照兩種數學等價的執行方式：

**Schedule A（GEMV 序列，等價於將 prefill 拆成 N 次 decode）**：

```
load W → compute y₁ → evict W
load W → compute y₂ → evict W      ← 重複載入同一個 W
       ⋮                              N 次
load W → compute y_N → evict W
```

總搬運：$N \cdot |W|$，共用潛力完全浪費。

**Schedule B（GEMM，prefill 的實際做法）**：

```
load W ─┬─ compute y₁
        ├─ compute y₂
        │       ⋮
        └─ compute y_N → evict W
```

總搬運：$|W|$，共用潛力完全兌現。

兩者數學結果一致——差別僅在 $W$ 進入 SRAM 後被**重用的次數**。GEMM kernel（cuBLAS、CUTLASS）的核心設計即是最大化此重用，透過 **tile decomposition** 將大矩陣切成能塞進 SRAM 的小塊：對每個 $W$ tile 載入 SRAM 後，**所有相關的 $X$ tile 依序流過完成乘加**，再 evict 換下一個 $W$ tile。tile 大小由兩個條件夾出：

- **夠大**：使 $W$ tile 載入後能服務足夠多 $X$ tile，攤平搬運成本；
- **夠小**：能塞進 SRAM 與暫存器，並讓 Tensor Core 維持滿載。

這直接解釋了前述算術強度公式 $I = 2N/b$ 的形成機制——分母由 $W$ tile 的單次載入決定，分子由 N 倍 reuse 撐起。同時也預示了下一節將探討的對偶困境：**當 N 退化為 1，就沒有任何 reuse 可言**，每次 $W$ 載入只能服務一個 token，搬運成本完全暴露在頻寬上。

---

## 4. 為什麼 Decode 是 Memory-Bound

### 4.1 「位置改變」與權重的重複載入

**權重在推論時是唯讀的**——它的數值完全不變。需要釐清兩個常被混淆的概念：

| 概念 | 發生場景 | 意義 |
|---|---|---|
| 權重 update | 訓練時 | 反向傳播改變 W 數值 |
| 權重 movement | 推論時 | W 數值不動，只是搬運位置 |

那為什麼推論時權重要「不斷搬」？根本原因是 SRAM 容量遠小於模型——任一時刻，**計算焦點**只能停留在權重的一小部分上，焦點移動就必須重新載入：

**空間維度的位置改變**：
$$
\text{Embedding} \to \text{Layer}_0 \to \text{Layer}_1 \to \cdots \to \text{Layer}_{31} \to \text{LM Head}
$$
每進入下一層，前一層的權重必須從 SRAM evict 以騰出空間。

**時間維度的位置改變**：
$$
\text{token}_1: W_0 \to W_1 \to \cdots \to W_{31} \quad (\text{14 GB 流過一次})
$$
$$
\text{token}_2: W_0 \to W_1 \to \cdots \to W_{31} \quad (\text{14 GB 又流過一次})
$$

每生成一個 token 就是一次完整 forward pass。Decode N 個 token，總搬運量為 $N \cdot |W|$。

### 4.2 Llama-7B Decode 的端到端頻寬計算

在 H100 上跑 Llama-7B (FP16, 4K context) decode：

```
每 token 搬運量：
  權重         : 14 GB
  KV cache 讀取: 2 GB  (隨 context 增長)
  Activation   : ~MB 級 (可忽略)
─────────────────────────
總計          : ~16 GB / token

理論上限 = HBM 頻寬 / 每 token 搬運量
        = 3350 GB/s / 16 GB
        ≈ 209 tokens/s
```

實測值通常落在 100~150 tokens/s（受 kernel launch overhead、cache miss 等影響），**確認 decode 已逼近頻寬上限**。再大的算力都沒用——硬體上 989 TFLOPS 中真正用到的不到 1%。

### 4.3 為什麼 batch=1 是最壞情況

若將 B 個獨立 user 的 decode 合併成 batch：
- 權重只需搬一次，被 B 個 user 共用
- 算術強度提升 B 倍

當 $B \geq I^* \approx 300$ 時，decode 也能進入 compute-bound 區域。這就是 **continuous batching** (vLLM、TGI 等) 的理論基礎。

### 4.4 自迴歸依賴：Decode 無法自我 Batch 的根本原因

前述分析顯示權重共用是模型結構天生賦予的，與 batching 無關。那麼自然會問：**既然 decode 接下來要生成的 K 個 token 都會用到同一組 $W$，為何不能像 prefill 那樣一次塞入？**

根本答案是：**未來的 token 尚未存在**。Decoder-only Transformer 的生成過程具有嚴格的自迴歸結構：

$$
t_{N+1} = \text{sample}\bigl(\text{model}(t_1, \ldots, t_N)\bigr)
$$

$$
t_{N+2} = \text{sample}\bigl(\text{model}(t_1, \ldots, t_N, t_{N+1})\bigr)
$$

$t_{N+2}$ 的輸入需要 $t_{N+1}$ 的具體值，而 $t_{N+1}$ 必須先走完整個模型才能被採樣出來。這條依賴鏈在硬體層鎖住了「將未來 K 個 token 一次餵入 $W$」的可能性。

此外，即使僅考慮生成單一 token 的過程，也存在**層間的權重輪替依賴**：

$$
t_{N+1} \text{ 的計算路徑}: \quad W_0 \to W_1 \to W_2 \to \cdots \to W_{31} \to \text{sample}
$$

從 $W_0$ 用完到下一個 token 再次需要 $W_0$，期間 SRAM 必須依序載入 $W_1, \ldots, W_{31}$ 共約 14 GB 的權重。$W_0$ 不可能在這段窗口內被保留——它必然遭後續權重 evict。

綜合起來，decode 階段的根本困境可表述為：

> **跨 token 的權重共用潛力確實存在，但被自迴歸的序列依賴與有限的 SRAM 容量共同鎖住，無法在硬體層自然兌現。**

現代 LLM 推論加速研究幾乎全部圍繞這個困境展開。主流策略可歸納為三類，差別在於從哪個維度「製造」更多能陪伴同一次 $W$ 載入的計算：

| 策略 | 機制 | 解鎖維度 |
|---|---|---|
| Speculative decoding | 以輕量 draft model 猜測未來 $K$ 個 token，由大模型一次驗證 | 時間軸（猜測未來） |
| Multi-token prediction (Medusa, EAGLE, DeepSeek-V3 MTP) | 模型結構上裝載多個輸出 head，一次預測未來 2–4 個 token | 結構軸（改寫自迴歸） |
| Continuous batching | 合併多個並發 user 的下一 token 共用一次權重載入 | 使用者軸（合併並發） |

三者本質一致——**製造更多可陪伴同一次權重載入的計算**——只是攻擊的維度不同。從 Roofline 觀點看，這些技術都在拉高 $I$，將 decode 由 memory-bound 推往 compute-bound 區域。

值得指出的是，若硬體能將整個模型常駐 SRAM（如 Cerebras WSE-3、Groq LPU），則 $W$ 在 SRAM 中可跨 token 持續存在，自迴歸依賴造成的搬運懲罰會從硬體層直接消失。這也是純 SRAM 架構能在 70B 模型上達到 500+ tokens/s decode 速度的原因——它不在演算法層繞過困境，而是從硬體前提層直接解除它。

---

## 5. 跨架構比較：GPU vs 邊緣 NPU

### 5.1 物理原理普世，工程取捨各異

馮紐曼瓶頸對所有架構都成立。但不同硬體會在「儲存—頻寬—計算」三角中選擇不同平衡點：

| 指標 | H100 GPU | 典型邊緣 NPU (如 Qualcomm IQ-9075 級) |
|---|---|---|
| 計算 (TOPS INT8) | ~2000 | 50~100 |
| 主記憶體 | HBM3, 80 GB | LPDDR5, 共享 4~16 GB |
| 主記憶體頻寬 | 3.35 TB/s | 50~100 GB/s |
| 片上 SRAM | ~50 MB | 4~32 MB |
| Ridge point $I^*$ | ~300 | ~500~1000 |
| 主要量化精度 | FP16 / FP8 | INT8 / INT4 |

### 5.2 邊緣 NPU 的優化策略源自相同原理

由於 LPDDR 頻寬遠低於 HBM，邊緣 NPU 必須更激進地降低搬運成本：

1. **量化到 INT4**：權重大小 ÷ 4 → 等效頻寬 × 4
2. **更大的片上 SRAM 比例**：讓部分層的權重常駐 SRAM
3. **Weight-stationary dataflow** (systolic array)：將權重「釘」在 PE 陣列裡，讓 activation 流過
4. **更積極的 KV cache 壓縮**：sliding window、INT4 KV、GQA

注意 (2)–(3) 在 GPU 上不適用，因為 GPU 採 SIMT 架構與較通用的 cache 機制；但兩者背後的目標一致：**減少跨晶片搬運次數**。

### 5.3 邊緣 LLM 的快速估算公式

對任一硬體，**忽略計算瓶頸時**的 decode 上限可估算為：

$$
\text{tokens/s} \lesssim \frac{\text{可用頻寬}}{|W|_{\text{quantized}} + |\text{KV}|}
$$

例：50 GB/s LPDDR 頻寬，跑 1B 模型 INT4 量化 (~0.5 GB)，2K context (~MB 級 KV)：

$$
\text{上限} \approx \frac{50}{0.5} = 100 \text{ tokens/s}
$$

實際可用值約為理論值的 50~70%。

---

## 6. 從原理推導的緩解策略

下表整理主流優化技術與其攻擊的 Roofline 維度：

| 技術 | 攻擊目標 | 機制 |
|---|---|---|
| Continuous batching | ↑ $I$ | 多 user 共用一次權重載入 |
| Speculative decoding | ↑ $I$ | 一次驗證多 draft tokens (偽 prefill) |
| Multi-token prediction (Medusa / MTP) | ↑ $I$ | 模型內建多輸出 head，單次 forward 產出多 token |
| 量化 (INT8/INT4) | ↓ 搬運量 | 縮小 $\|W\|$ 與 KV |
| GQA / MQA | ↓ 搬運量 | 縮減 KV head 數 |
| FlashAttention | ↓ 搬運量 | tile 化 attention 避免大 activation 寫回 |
| PagedAttention | 提升頻寬利用率 | 消除 KV cache 碎片 |
| Sliding window | ↓ KV 大小 | 截斷舊 KV |
| Tensor parallelism | ↑ 總頻寬 | 多 GPU 並聯 HBM |

注意所有有效策略都圍繞 Roofline 模型的兩個變數打轉：**要嘛拉高 $I$（讓硬體進入 compute-bound 區域），要嘛降低 $\|W\| + \|\text{KV}\|$（縮小頻寬負擔）**。這不是巧合——這是從第一性原理出發的必然結論。

---

## 7. 結語：未來的突破方向

如果頻寬瓶頸源自「儲存與計算的物理分離」這個馮紐曼前提，那麼真正的範式級突破必然要挑戰這個前提：

| 方向 | 範例 | 原理 |
|---|---|---|
| **Processing-in-Memory (PIM)** | Samsung HBM-PIM, SK Hynix AiM | 在 DRAM bank 內塞簡單運算單元，部分運算就地完成 |
| **Wafer-scale integration** | Cerebras WSE-3 | 整片晶圓當一顆晶片，40 GB 片上 SRAM，根本不用 HBM |
| **SRAM-first 設計** | Groq LPU | 全 SRAM 架構，犧牲容量換頻寬 |
| **3D 整合** | AMD MI300X (HBM3e + chiplet) | 縮短儲存到計算的物理距離 |

這些都是在挑戰「儲存與計算必須分離」這個假設。但在它們成熟之前，理解 Roofline 模型、算術強度、以及 prefill/decode 在頻寬上的不對稱性，仍是設計與優化 LLM 系統的基本功。

---

## 附錄 A：關鍵公式速查

**算術強度（線性層）**：
$$
I_{\text{linear}} = \frac{2N}{b}, \quad N = \text{batch} \times \text{seq\_len}
$$

**Roofline throughput**：
$$
T = \min(\text{FLOPs}_{\max}, \, I \times BW)
$$

**KV cache 大小**：
$$
|\text{KV}| = 2 \cdot L \cdot S \cdot d_{\text{model}} \cdot b
$$

**Decode token rate 上限**：
$$
\text{tokens/s} \leq \frac{BW}{|W| + |\text{KV}|}
$$

## 附錄 B：建議延伸閱讀

- Williams, Waterman, Patterson (2009). *Roofline: An Insightful Visual Performance Model*. CACM.
- Pope et al. (2022). *Efficiently Scaling Transformer Inference*. Google.
- Dao et al. (2022). *FlashAttention*. NeurIPS.
- Kwon et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM). SOSP.
- Leviathan et al. (2023). *Fast Inference from Transformers via Speculative Decoding*. ICML.
