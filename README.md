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

The **Spectral** layer mixes token representations in a binary-dilated way using per-token frequencies and per-level phases.

### 1. Token frequency projection

Each token is first projected into a per-head frequency space:

```python
theta = norm(self.freq_token(x))  # x: (B, T, dim)
theta = theta.view(B, T, self.num_heads, self.head_dim)
```

- `freq_token`: linear layer over the embedding dimension.
- `norm(...)`: normalization before mixing.
- Result is reshaped into `(batch, time, heads, head_dim)`.

---

### 2. Per-level phase generation

We generate a phase value for each token, level, and head:

```python
phi = self.freq_level(x)          # (B, T, max_lvls * num_heads)
phi = phi.view(B, T, self.max_lvls, self.num_heads)
```

- `max_lvls` is based on the sequence length: `(sequence_len - 1).bit_length()`.
- Each level corresponds to a different binary-dilated step.

---

### 3. Binary-dilated mixing

For each level `l`, we mix the current state with a time-shifted version of itself:

```python
def _mix(self, theta, phi_l, step):
    # theta: (B, T, H, Dh)
    # phi_l: (B, T, H, 1)

    theta_prev = torch.zeros_like(theta)
    if step < T:
        theta_prev[:, step:] = theta[:, :-step]

    cos = torch.cos(phi_l)
    sin = torch.sin(phi_l)

    out = cos * theta + sin * theta_prev
    return out * (1 / math.sqrt(2.0))
```

The loop over levels is checkpointed for memory efficiency:

```python
theta = checkpoint(self._loop, theta, phi, use_reentrant=False)
```

---

### 4. Final projection

After all levels are applied, we project back to the model dimension:

```python
theta = theta.view(B, T, self.dim)
y = self.out_proj(theta)          # (B, T, dim)
```

- `out_proj`: final linear layer over the embedding dimension.
- `y` is the output of the Spectral layer.


[View Spectral implementation (line 41)](https://github.com/mctomi/spectral/blob/master/nanochat/gpt.py#L41)

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
