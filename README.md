# Mixture-of-Agents Enhances Large Language Model Capabilities

This is a fork of https://github.com/togethercomputer/MoA with some tweaks to make it work with local models.

100% of the credit goes to the original authors.

---

<div align="center">
  <img src="assets/moa.jpg" alt="moa" style="width: 100%; display: block; margin-left: auto; margin-right: auto;" />
  <br>
</div>

Mixture of Agents (MoA) is a novel approach that leverages the collective strengths of multiple LLMs to enhance performance, achieving state-of-the-art results. By employing a layered architecture where each layer comprises several LLM agents, MoA significantly outperforms GPT-4 Omni's 57.5% on AlpacaEval 2.0 with a score of 65.1%, using only open-source models!

## Interactive Demo

We first present an interactive demo. It showcases a simple multi-turn chatbot where the final response is aggregated from various reference models.

### Setup

1. Setup your environment:

   ```shell
   cp .env.example .env
   vi .env
   ```

2. Install Requirements:

   ```shell
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

### Running the Demo

To run the interactive demo, execute the following script with Python:

```shell
python bot.py
```

The script will prompt you to input instructions interactively. Here's how to use it:

1. Start by entering your instruction at the ">>>" prompt.
2. The system will process your input using the predefined reference models.
3. It will generate a response based on the aggregated outputs from these models.
4. You can continue the conversation by inputting more instructions, with the system maintaining the context of the multi-turn interaction.
5. enter `exit` to exit the chatbot.

### Configuration

You can configure the demo by specifying the following parameters:

- `--aggregator`: The primary model used for final response generation.
- `--reference_models`: List of models used as references.
- `--temperature`: Controls the randomness of the response generation.
- `--max_tokens`: Maximum number of tokens in the response.
- `--rounds`: Number of rounds to process the input for refinement. (num rounds == num of MoA layers - 1)
- `--num_proc`: Number of processes to run in parallel for faster execution.
- `--multi_turn`: Boolean to toggle multi-turn interaction capability.

## Credit / Authors / Acknowledgements

Please see https://github.com/togethercomputer/MoA/
