# Spectral Mixer — A New O(T log T) Attention-Free Architecture

**Spectral** is a new sequence-mixing architecture built around logarithmic-depth hierarchical rotations.  
It offers global token interaction in **O(T log T)** time using only elementwise operations, enabling efficient long-context language modeling without attention.

Unlike conventional mechanisms, Spectral introduces a structured mixing process based on **binary‑dilated shifts** and **learned per‑head phase rotations**, providing a simple yet expressive alternative to attention for autoregressive models.

---

## Method

Tokens are projected into a low‑rank, multi‑head spectral space.  
For each level \( l = 0 \dots \lceil \log_2 T
ceil - 1 \), the sequence is shifted by \( 2^l \) positions and updated via a learned rotation:

\[
heta*t \leftarrow \cos(\phi*{l,t}) heta*t + \sin(\phi*{l,t}) heta\_{t - 2^l}.
\]

Implementation line:

```python
theta = cos * theta + sin * theta_prev
```

After all levels, states are merged and projected back to the model dimension.

---

## Core Properties

- **New architecture** based on hierarchical frequency‑modulated mixing
- **Global reach in log₂(T) steps**
- **No attention matrices or pairwise interactions**
- **Linear memory usage**
- **Efficient for long context windows**
- **Independent per‑head oscillators**

---

## Complexity

- **Time:** O(T log T)
- **Memory:** O(T)

---

## Status

An experimental architecture exploring attention‑free sequence modeling using rotational dynamics and logarithmic-depth propagation.

---

## License

The original code remains under the MIT License (see LICENSE).

All new contributions by Mateus Costamilan Tomiello are provided under the MIT License with
the Commons Clause restriction (see LICENSE-ADDITIONS), meaning they cannot be
sold or used commercially without permission.
