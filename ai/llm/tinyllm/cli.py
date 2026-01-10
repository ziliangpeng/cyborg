"""Interactive CLI for TinyLLM."""

import time

import click
from tinygrad import Tensor

from ai.llm.tinyllm import GPT2, OPT, generate
from ai.llm.tinyllm.models import BaseModel
from ai.llm.tinyllm.utils import Tokenizer


def load_model(model_name: str) -> BaseModel:
    """Load model by name."""
    if model_name.startswith("facebook/opt-"):
        return OPT.from_pretrained(model_name)
    else:
        return GPT2.from_pretrained(model_name)


def format_param_count(count: int) -> str:
    """Format parameter count with appropriate suffix."""
    if count >= 1e9:
        return f"{count / 1e9:.1f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.0f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.0f}K"
    return str(count)


def run_prompt(llm: BaseModel, tokenizer: Tokenizer, prompt: str, max_tokens: int, temperature: float, top_k: int | None, sample: bool) -> None:
    """Run a single prompt through the model and print results."""
    # Encode prompt
    input_tokens = tokenizer.encode(prompt)
    input_ids = Tensor([input_tokens])
    num_input_tokens = len(input_tokens)

    # Generate
    gen_start = time.perf_counter()
    output_ids = generate(
        llm,
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        do_sample=sample,
    )
    gen_time = time.perf_counter() - gen_start

    # Decode output
    output_tokens = output_ids[0].numpy().tolist()
    num_output_tokens = len(output_tokens) - num_input_tokens
    output_text = tokenizer.decode(output_tokens)

    # Print output
    click.echo()
    click.echo(output_text)
    click.echo()

    # Print stats
    tokens_per_sec = num_output_tokens / gen_time if gen_time > 0 else 0
    click.echo("Stats:")
    click.echo(f"  Input tokens:  {num_input_tokens}")
    click.echo(f"  Output tokens: {num_output_tokens}")
    click.echo(f"  Generation:    {gen_time:.3f}s")
    click.echo(f"  Tokens/sec:    {tokens_per_sec:.1f}")


@click.command()
@click.option("--model", default="gpt2", help="Model name to use")
@click.option("--max-tokens", default=50, help="Maximum tokens to generate")
@click.option("--temperature", default=1.0, help="Sampling temperature")
@click.option("--top-k", default=None, type=int, help="Top-k sampling")
@click.option("--sample/--greedy", default=True, help="Use sampling or greedy decoding")
@click.option("--prompt", default=None, help="Prompt to run (non-interactive mode)")
def main(model: str, max_tokens: int, temperature: float, top_k: int | None, sample: bool, prompt: str | None):
    """Interactive CLI for TinyLLM inference."""
    click.echo(f"Loading model: {model}")
    load_start = time.perf_counter()
    llm = load_model(model)
    tokenizer = Tokenizer.for_model(model)
    load_time = time.perf_counter() - load_start
    click.echo(f"Model loaded in {load_time:.2f}s ({format_param_count(llm.param_count())} params)")

    # Non-interactive mode: run single prompt and exit
    if prompt is not None:
        run_prompt(llm, tokenizer, prompt, max_tokens, temperature, top_k, sample)
        return

    # Interactive mode
    click.echo()
    click.echo("TinyLLM Interactive CLI")
    click.echo("Type 'quit' or Ctrl+C to exit.")
    click.echo()

    try:
        while True:
            user_prompt = input("> ")
            if user_prompt.lower() in ("quit", "exit"):
                break
            if not user_prompt.strip():
                continue

            run_prompt(llm, tokenizer, user_prompt, max_tokens, temperature, top_k, sample)
            click.echo()

    except KeyboardInterrupt:
        click.echo("\nExiting...")


if __name__ == "__main__":
    main()
