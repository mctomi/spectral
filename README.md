# Spectral Mixer — Attention-Free O(T log T) Architecture

Spectral is a new sequence-mixing architecture that replaces attention with a lightweight, logarithmic-depth rotational mixer.  
It enables efficient long-context modeling while keeping implementation simple and fully compatible with standard autoregressive LLM blocks.

---

## Key Features

- **New attention-free architecture**
- **O(T log T)** mixing depth  
- **Linear memory usage**
- **Global token interaction in log₂(T) steps**
- **Binary‑dilated shifts:** 1, 2, 4, 8, ...
- **Learned per-head phase rotations**
- **Low-rank spectral space for efficiency**
- **Pure PyTorch, no custom kernels**
- **Drop-in replacement for Transformer attention**

---

## How It Works (Code-Oriented)

### 1. Low-rank projection

```python
theta = freq_token(x)          # (B,T,rank)
theta = theta.view(B,T,H,Dh)   # split into heads
```

### 2. Per-level phase generation

```python
phi = freq_level(x)
phi = phi.view(B,T,L,H,Dh)     # L = log2(T)
```

### 3. Binary-dilated mixing loop

```python
for l in range(L):
    step = 1 << l              # 1,2,4,8,...
    phi_l = phi[:,:,l]
    theta = cos(phi_l)*theta + sin(phi_l)*theta_prev
```

`theta_prev` is the shifted sequence:

```python
theta_prev[:, step:] = theta[:, :-step]
```

### 4. Merge and project back

```python
theta = theta.view(B,T,rank)
out   = out_proj(theta)
```

---

## Capabilities

- Handles very long sequences without quadratic cost  
- Simple mechanism with predictable behavior  
- Provides hierarchical, multi-scale mixing  
- Works with any embedding size or head layout  
- Requires only standard PyTorch ops  
- Effective drop-in module for experimentation

---

## Status

Spectral is experimental and intended for research into alternative sequence-processing architectures.

---

## License

The original code remains under the MIT License (see LICENSE).

All new contributions by Mateus Costamilan Tomiello are provided under the MIT License with
the Commons Clause restriction (see LICENSE-ADDITIONS), meaning they cannot be
sold or used commercially without permission.
