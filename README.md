# Multi-Programming Language Evaluation of Large Language Models of Code (MultiPL-E)

**New**: For a more challenging multi-language benchmark, check out [Ag-LiveCodeBench-X](https://github.com/nuprl/Ag-LiveCodeBench-X)
and its accompanying paper, [Agnostics](https://arxiv.org/abs/2508.04865).

## Introduction

MultiPL-E is a system for translating unit test-driven neural code generation
benchmarks to new languages. We have used MultiPL-E to translate two popular
Python benchmarks (HumanEval and MBPP) to 18 other programming languages.

For more information:

- MultiPL-E is part of the [BigCode Code Generation LM Harness]. This
  is the easiest way to use MultiPL-E.
- The [Multilingual Code Models Evaluation] by BigCode evaluates Code LLMs
  using several benchmarks, including MultiPL-E.
- Read our paper [MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code Generation].
- The [MultiPL-E dataset] of translated prompts is available on the Hugging Face
  Hub.

## Tutorial

These are instructions on how to use MultiPl-E directly, without the 
BigCode evaluation harness.

In this tutorial, we will run a small experiment to evaluate the performance of
[SantaCoder] on Rust with a small subset of the MBPP benchmarks. 
We will only fetch 20 completions per problem, so that you
can run it quickly on a single machine.  
You can also run on the full suite of benchmarks or substitute your own
benchmark programs. Later, we'll show you how to add support for other languages
and evaluate other models.

### Prerequisites

1. Check out the repository:    

   ```bash
   git clone https://github.com/nuprl/MultiPL-E
   ```

2. Enter the repository directory:

   ```bash
   cd MultiPL-E
   ```

3. You will need Python 3.8 or higher.

4. You will need to install some Python packages:

    ```bash
    pip3 install -r requirements.txt
    ```

3. You need to install [Docker].



### Background

We first finetune the model using Trainer. After each epoch, each task and at the end of finetuning, we validate the model. Validation using MultiPL-E is a two step process:

1. We *generate* completions, which requires a GPU.

2. We *execute* the generated completions, which requires a machine that
   supports Docker or Podman.

3. We *calculate Pass@k* based on the results of the tests.

### Set up container

When you first run evaluation, you need to pull and tag the [execution container](https://github.com/nuprl/MultiPL-E/pkgs/container/multipl-e-evaluation):


```bash
docker pull ghcr.io/nuprl/multipl-e-evaluation
docker tag ghcr.io/nuprl/multipl-e-evaluation multipl-e-eval
```

### Fine-tune model
Run the following command. Check out `main.py` for more details on all arguments:

```bash
python main.py \
  --task_list c_sharp cpp \
  --log_filepath logs/test.log 
```


## Credits

MultiPL-E was originally authored by:

- Federico Cassano (Northeastern University)
- John Gouwar (Northeastern University)
- Daniel Nguyen (Hanover High School)
- Sydney Nguyen (Wellesley College)
- Luna Phipps-Costin (Northeastern University)
- Donald Pinckney (Northeastern University)
- Ming-Ho Yee (Northeastern University)
- Yangtian Zi (Northeastern University)
- Carolyn Jane Anderson (Wellesley College)
- Molly Q Feldman (Oberlin College)
- Arjun Guha (Northeastern University and Roblox Research)
- Michael Greenberg (Stevens Institute of Technology)
- Abhinav Jangda (University of Massachusetts Amherst)

We thank Steven Holtzen for loaning us his GPUs for a few weeks. We thank
[Research Computing at Northeastern University] for supporting the
Discovery cluster.

Several people have since contributed to MultiPL-E. Please see the
[changelog](https://huggingface.co/datasets/nuprl/MultiPL-E) for those acknowledgments.

[BigCode Code Generation LM Harness]: https://github.com/bigcode-project/bigcode-evaluation-harness
[MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code Generation]: https://ieeexplore.ieee.org/abstract/document/10103177
[SantaCoder]: https://arxiv.org/abs/2301.03988
[MultiPL-E dataset]: https://huggingface.co/datasets/nuprl/MultiPL-E
[StarCoder]: https://arxiv.org/abs/2305.06161
[Multilingual Code Models Evaluation]: https://huggingface.co/spaces/bigcode/multilingual-code-evals
[Conda]: https://conda.io/
[Podman]: https://podman.io/
[Docker]: https://www.docker.com/
