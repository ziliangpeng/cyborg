"""Tests for OPT model implementation."""

import time

from tinygrad import Tensor

from ai.llm.tinyllm import OPT, OPTConfig, generate


def test_config():
    """Test configuration dataclass."""
    config = OPTConfig.opt_125m()
    assert config.n_layer == 12
    assert config.n_head == 12
    assert config.n_embd == 768
    assert config.vocab_size == 50272
    assert config.n_positions == 2048
    assert config.n_inner == 3072
    assert config.position_offset == 2
    print("Config test passed")

    # Test OPT-350m config with word_embed_proj_dim
    config_350m = OPTConfig.opt_350m()
    assert config_350m.word_embed_proj_dim == 512
    print("Config 350m test passed")


def test_forward_pass():
    """Test model forward pass with random weights."""
    config = OPTConfig.opt_125m()
    model = OPT(config)

    # Random input
    batch_size, seq_len = 2, 10
    input_ids = Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])

    # Forward pass
    logits = model(input_ids)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    print(f"Forward pass test passed: output shape {logits.shape}")


def test_pretrained_loading():
    """Test loading pre-trained OPT weights."""
    print("Loading OPT-125M from HuggingFace...")
    start = time.time()
    model = OPT.from_pretrained("facebook/opt-125m")
    load_time = time.time() - start
    print(f"Loaded in {load_time:.2f}s")

    # Verify a simple forward pass
    # Token IDs for "The world is" (using OPT tokenizer IDs)
    # OPT uses </s> as BOS which has ID 2
    input_ids = Tensor([[133, 232, 16]])  # "The world is" in OPT tokenizer
    logits = model(input_ids)

    assert logits.shape == (1, 3, 50272)
    print("Pretrained loading test passed")

    # Check that logits are reasonable (not all zeros or NaN)
    logits_np = logits.numpy()
    assert not (logits_np == 0).all(), "Logits are all zeros"
    assert not any(map(lambda x: x != x, logits_np.flatten())), "Logits contain NaN"
    print("Logits sanity check passed")


def test_generation_greedy():
    """Test greedy text generation."""
    model = OPT.from_pretrained("facebook/opt-125m")

    # "The" token in OPT
    input_ids = Tensor([[133]])

    # Greedy generation
    output = generate(model, input_ids, max_new_tokens=5, do_sample=False)
    assert output.shape[1] == 6  # 1 input + 5 generated
    print(f"Greedy generation test passed: generated {output.shape[1]} tokens")
    print(f"Token IDs: {output.numpy()}")


def test_generation_sampling():
    """Test sampling-based text generation."""
    model = OPT.from_pretrained("facebook/opt-125m")

    # "The" token in OPT
    input_ids = Tensor([[133]])

    # Sampling generation with temperature
    output = generate(model, input_ids, max_new_tokens=5, do_sample=True, temperature=0.8)
    assert output.shape[1] == 6  # 1 input + 5 generated
    print(f"Sampling generation test passed: generated {output.shape[1]} tokens")
    print(f"Token IDs: {output.numpy()}")


def test_generation_top_k():
    """Test top-k sampling."""
    model = OPT.from_pretrained("facebook/opt-125m")

    # "The" token in OPT
    input_ids = Tensor([[133]])

    # Top-k sampling
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
