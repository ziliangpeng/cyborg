"""Text generation utilities for TinyLLM."""

from typing import Optional

import numpy as np

from tinygrad import Tensor


def generate(
    model,
    input_ids: Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    do_sample: bool = False,
) -> Tensor:
    """
    Generate text tokens autoregressively.

    Args:
        model: GPT2 model instance
        input_ids: (batch_size, seq_len) initial token IDs
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (1.0 = neutral, <1.0 = sharper, >1.0 = flatter)
        top_k: If set, only sample from top-k most likely tokens
        do_sample: If True, sample from distribution; if False, greedy decoding

    Returns:
        (batch_size, seq_len + max_new_tokens) generated token IDs
    """
    for _ in range(max_new_tokens):
        # Truncate to max sequence length if needed
        seq_len = input_ids.shape[1]
        if seq_len > model.config.n_positions:
            input_ids = input_ids[:, -model.config.n_positions :]

        # Get logits for next token
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]  # (batch, vocab_size)

        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Sample or greedy decode
        if do_sample:
            # Optional top-k filtering
            if top_k is not None:
                next_token_logits = _top_k_filtering(next_token_logits, top_k)

            # Sample from softmax distribution
            probs = next_token_logits.softmax(axis=-1)
            next_token = _multinomial_sample(probs)
        else:
            # Greedy: take argmax
            next_token = next_token_logits.argmax(axis=-1, keepdim=True)

        # Append to sequence
        input_ids = Tensor.cat(input_ids, next_token, dim=1)

    return input_ids


def _top_k_filtering(logits: Tensor, k: int) -> Tensor:
    """
    Zero out all logits except the top-k.

    Args:
        logits: (batch_size, vocab_size) logits
        k: Number of top tokens to keep

    Returns:
        Filtered logits with non-top-k set to -inf
    """
    values = logits.numpy()
    # Get k-th largest value for each batch
    kth_values = np.partition(values, -k, axis=-1)[:, -k : -k + 1]

    # Create mask: True where logits are below threshold
    threshold = Tensor(kth_values)
    mask = logits < threshold

    # Set masked values to -inf
    return logits * (1 - mask.float()) + mask.float() * (-1e10)


def _multinomial_sample(probs: Tensor) -> Tensor:
    """
    Sample from probability distribution.

    Args:
        probs: (batch_size, vocab_size) probability distribution

    Returns:
        (batch_size, 1) sampled token indices
    """
    probs_np = probs.numpy()

    samples = []
    for p in probs_np:
        # Normalize to handle any numerical issues
        p = p / p.sum()
        idx = np.random.choice(len(p), p=p)
        samples.append(idx)

    return Tensor(np.array(samples).reshape(-1, 1))
