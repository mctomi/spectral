import torch
import torch.nn.functional as F
from collections import deque

# -----------------------------------------------------------
# Helpers
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / max(temperature, 1e-6)

    if top_k is not None:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, 1, generator=rng)
        return idx.gather(1, choice)
    else:
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1, generator=rng)

# -----------------------------------------------------------
class RowState:
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []
        self.forced_tokens = deque()
        self.in_python_block = False
        self.python_expr_tokens = []
        self.completed = False

# -----------------------------------------------------------
class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None,
                 temperature=1.0, top_k=None, seed=42):

        assert isinstance(tokens, list) and isinstance(tokens[0], int)

        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Special tokens
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        # state for each row
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        num_generated = 0

        while True:
            # stopping conditions
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(s.completed for s in row_states):
                break

            token_column = []
            token_masks = []

            for state in row_states:
                # decide next token (forced or sampled)
                if len(state.forced_tokens) > 0:
                    next_token = state.forced_tokens.popleft()
                    token_masks.append(0)
                    token_column.append(next_token)
                else:
                    # prepare input: full history!
                    ids = torch.tensor([state.current_tokens], dtype=torch.long, device=device)
                    logits = self.model(ids)[:, -1, :]
                    next_token = sample_next_token(logits, rng, temperature, top_k).item()
                    token_masks.append(1)
                    token_column.append(next_token)

                # update row state
                state.current_tokens.append(next_token)

                if next_token == assistant_end or next_token == bos:
                    state.completed = True

                # python tool logic
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    expr = self.tokenizer.decode(state.python_expr_tokens)
                    result = self._safe_eval(expr)
                    if result is not None:
                        result_tokens = self.tokenizer.encode(str(result))
                        state.forced_tokens.append(output_start)
                        for t in result_tokens:
                            state.forced_tokens.append(t)
                        state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            yield token_column, token_masks
            num_generated += 1

    # simplified version of your safe calculator
    def _safe_eval(self, expr):
        try:
            return eval(expr, {"__builtins__": {}})
        except:
            return None

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[1] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples

        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token in (assistant_end, bos):
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)

            if all(completed):
                break

        return results, masks
