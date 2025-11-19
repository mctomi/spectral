# Spectral Mixer — A Minimal O(T log T) Attention-Free Architecture

Spectral is a lightweight sequence-mixing module designed to replace attention with a simple, logarithmic‑depth rotational mixer.  
It supports efficient long‑context processing while maintaining a clean, PyTorch‑only implementation compatible with standard autoregressive LLM blocks.

---

## Key Features

- Attention‑free architecture  
- O(T log T) mixing depth  
- Linear memory footprint  
- Global token interaction in log₂(T) steps  
- Binary‑dilated shifts (1, 2, 4, 8, …)  
- Learned per‑head phase rotations  
- Low‑rank spectral representation  
- No custom kernels  
- Drop‑in replacement for attention layers  

---

## How It Works

### 1. Low‑rank projection

```python
theta = freq_token(x)
theta = theta.view(B, T, H, Dh)
```

### 2. Per‑level phase generation

```python
phi = freq_level(x)
phi = phi.view(B, T, L, H, Dh)
```

### 3. Binary‑dilated mixing

```python
for l in range(L):
    step = 1 << l
    phi_l = phi[:, :, l]
    theta = cos(phi_l) * theta + sin(phi_l) * theta_prev
```

Shifted state:

```python
theta_prev[:, step:] = theta[:, :-step]
```

### 4. Final projection

```python
theta = theta.view(B, T, rank)
out = out_proj(theta)
```

[View Spectral implementation (line 42)](https://github.com/mctomi/spectral/blob/master/nanochat/gpt.py#L42)

---

## Capabilities

- Efficient for long sequences without quadratic cost  
- Predictable, stable mixing behavior  
- Multi‑scale hierarchical propagation  
- Compatible with any head configuration  
- Pure PyTorch implementation  
- Suitable for rapid experimentation  

---

## Status

Spectral is experimental and intended for research on alternative sequence‑processing mechanisms.

---

## License

Original NanoChat code: MIT License (see LICENSE).  
New contributions by Mateus Costamilan Tomiello: MIT License with Commons Clause (see LICENSE‑ADDITIONS), restricting commercial use without permission.
