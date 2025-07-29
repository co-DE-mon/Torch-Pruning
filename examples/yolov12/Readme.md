
# YOLO-v12 Pruning Project

## 🧩 Project Overview

**🎯 Objective:**  
Build a lighter, faster YOLO detector through a two-pronged approach:

### Model Compression
- Apply **structured channel pruning** using `torch_pruning.GroupNormPruner`:
  - Progressive strategy:
    - Multiple small pruning iterations  
    - Fine-tuning after each step  
    - Repeat  
- **Target:**
  - ≥ **reduction** in MACs and parameters  
  - **No to little drop in detection accuracy**
  - Run efficiently on standard hardware (i.e., **no need for sparse computation support**)

---

## 🚧 Current Challenge

### ❗ Core Issue: **Pruning Compatibility**

- **Status:**  
  - Pruning pipeline **works** when `A2C2f` is excluded via `ignored_layers`.
  - **Hangs indefinitely** when attempting to prune through `A2C2f`.

---

## 🐛 Symptom

```python
# ✅ This works (but defeats the purpose of pruning A2C2f):
ignored_layers = []
unwrapped_parameters = []

for m in model.model.modules():
    if isinstance(m, (Detect,)):
        ignored_layers.append(m)
    elif isinstance(m, A2C2f):  # Skip pruning A2C2f
        ignored_layers.append(m)

# ❌ This hangs (when pruning through A2C2f):
ignored_layers = []  # Do not ignore A2C2f
pruner = GroupNormPruner(..., ignored_layers=ignored_layers)  # Hangs here
```

---

## 🔬 Possible Cause Analysis

### ⚠️ Pruner Graph Construction Hangs Due To:

#### 1. **Position Encoding Layer Channel Mismatch (MAJOR CAUSE in my opinion)**
```python
self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)
```
- **Issue:** `all_head_dim ≠ dim` → causes **ambiguous channel lineage**
- GroupNormPruner cannot trace dependency **across grouped convolutions**

---

#### 2. **Complex Tensor Reshaping Operations**
- Dynamic reshape based on `area` parameter:
```python
qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
```
- Additional operations:
  - `permute`
  - `view`
  - `flatten`
- **Impact:** These are **raw tensor ops**, **invisible to module-based pruning graph** (not tracked by dependency graph builder)

---

#### 3. **Matrix Multiplication-Based Attention**
- Core attention uses `matmul`:
```python
attn = (q @ k.transpose(-2, -1)) * scale
```
- **Issue:** `matmul` dependencies **not handled** by `torch_pruning` dependency tracker  
- Leads to **incomplete graph** and eventual hang

---

### ✅ Unwrapped Parameters
- Original implementation used raw `nn.Parameter` for **gamma scaling**  
  (i.e., parameters **not wrapped** in `nn.Module` → untracked)
- **Fix:** Converted to **depth-wise convolution**:
```python
self.gamma = Conv(c2, c2, 1, g=c2, act=False)
```
- Ensures `gamma` scaling is **trackable by pruner**

---

