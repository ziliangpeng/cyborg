"""Tests for LLaMA model implementation."""

import time

from tinygrad import Tensor

from ai.llm.tinyllm import LLaMA, LlamaConfig, generate


def test_config():
    """Test LlamaConfig fields."""
    config = LlamaConfig.open_llama_3b()
    assert config.n_layer == 26
    assert config.n_head == 32
    assert config.n_embd == 3200
    assert config.n_inner == 8640
    assert config.vocab_size == 32000
    assert config.n_positions == 2048
    assert config.rms_norm_eps == 1e-6
    assert config.rope_theta == 10000.0
    print("Config 3B test passed")

    config_7b = LlamaConfig.open_llama_7b()
    assert config_7b.n_layer == 32
    assert config_7b.n_head == 32
    assert config_7b.n_embd == 4096
    assert config_7b.n_inner == 11008
    print("Config 7B test passed")

    config_13b = LlamaConfig.open_llama_13b()
    assert config_13b.n_layer == 40
    assert config_13b.n_head == 40
    assert config_13b.n_embd == 5120
    assert config_13b.n_inner == 13824
    print("Config 13B test passed")


def test_forward_pass():
    """Test model forward pass with random weights."""
    config = LlamaConfig.open_llama_3b()
    model = LLaMA(config)

    batch_size, seq_len = 2, 10
    input_ids = Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])

    logits = model(input_ids)

    assert logits.shape == (batch_size, seq_len, config.vocab_size), (
        f"Expected {(batch_size, seq_len, config.vocab_size)}, got {logits.shape}"
    )
    print(f"Forward pass test passed: output shape {logits.shape}")


def test_pretrained_loading():
    """Test loading pre-trained Open LLaMA 3B weights."""
    print("Loading openlm-research/open_llama_3b from HuggingFace...")
    start = time.time()
    model = LLaMA.from_pretrained("openlm-research/open_llama_3b")
    load_time = time.time() - start
    print(f"Loaded in {load_time:.2f}s")

    input_ids = Tensor([[1, 9038, 2501, 263, 931]])  # "Once upon a time" approx
    logits = model(input_ids)

    assert logits.shape == (1, 5, 32000)
    print("Pretrained loading test passed")


def test_generation_greedy():
    """Test greedy text generation."""
    model = LLaMA.from_pretrained("openlm-research/open_llama_3b")

    input_ids = Tensor([[1]])  # BOS token

    output = generate(model, input_ids, max_new_tokens=5, do_sample=False)
    assert output.shape[1] == 6  # 1 input + 5 generated
    print(f"Greedy generation test passed: generated {output.shape[1]} tokens")
    print(f"Token IDs: {output.numpy()}")


def test_generation_sampling():
    """Test sampling-based text generation."""
    model = LLaMA.from_pretrained("openlm-research/open_llama_3b")

    input_ids = Tensor([[1]])  # BOS token

    output = generate(model, input_ids, max_new_tokens=5, do_sample=True, temperature=0.8)
    assert output.shape[1] == 6  # 1 input + 5 generated
    print(f"Sampling generation test passed: generated {output.shape[1]} tokens")
    print(f"Token IDs: {output.numpy()}")


def test_generation_top_k():
    """Test top-k sampling."""
    model = LLaMA.from_pretrained("openlm-research/open_llama_3b")

    input_ids = Tensor([[1]])  # BOS token

    output = generate(model, input_ids, max_new_tokens=5, do_sample=True, top_k=50, temperature=0.8)
    assert output.shape[1] == 6  # 1 input + 5 generated
    print(f"Top-k sampling test passed: generated {output.shape[1]} tokens")
    print(f"Token IDs: {output.numpy()}")


if __name__ == "__main__":
    test_config()
    test_forward_pass()
    test_pretrained_loading()
    test_generation_greedy()
    test_generation_sampling()
    test_generation_top_k()
    print("\nAll tests passed!")
