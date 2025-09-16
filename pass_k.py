"""

This script calculates pass@k. It receives a list of directories as its
argument, and calculates the mean pass@k for the set of problems in each
directory. It checks that all results in a directory were generated at the same
temperature. It calculates pass@1 for temperature 0.2 and both pass@10 and
pass@100 for temperature 0.8.

The output has the following columns:

- Dataset: the name of a directory
- Pass@k: the value of k
- Estimate: the mean pass@k for the problems in the directory
- NumProblems: the number of problems in the directory
- MinCompletions: the minimum number of completions for any problem in the 
  directory
- MaxCompletions: the maximum number of completions for any problem in the
  directory
"""
import numpy as np
from pathlib import Path
import itertools
import argparse
import json
from multipl_e.util import gunzip_json, eprint


def estimator(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def for_file(path: Path, ks: list[int]):
    if path.suffix == ".gz":
        data = gunzip_json(path)
    else:
        with open(path, 'r') as f:
            data = json.load(f)

    if data is None:
        return None
    n = len(data["results"])
    c = len([True for r in data["results"] if r["status"]
            == "OK" and r["exit_code"] == 0])
    output_dict = {}
    for k in ks:
        output_dict[f"pass@{k}"] = estimator(n, c, k)
    output_dict["n"] = n,
    output_dict["c"] = c,
    output_dict["temperature"] = data["temperature"] if "temperature" in data else 0.2
    return output_dict

def calculate_pass_k(args):
    if not args.suppress_header:
        print("Dataset,Pass@k,Estimate,NumProblems,MinCompletions,MaxCompletions")
    if args.k is None or args.k == 1:
        ks = [1]
    else:
        ks = [1, args.k]
    for d in args.dirs:
        results = [for_file(p, ks) for p in itertools.chain(
            Path(d).glob("*.results.json"), Path(d).glob("*.results.json.gz"))]
        results = [r for r in results if r is not None]
        name = d.split("/")[-1] if d.split("/")[-1] != "" else d.split("/")[-2]
        temperatures = set(r["temperature"] for r in results)
        if len(temperatures) != 1:
            eprint(
                f"Found multiple temperatures {temperatures} in {d} {results}")
            continue
        temperature = list(temperatures)[0]
        num_problems = len(results)
        min_completions = np.min([r["n"] for r in results])
        max_completions = np.max([r["n"] for r in results])

        output = []
        for k in ks:
            pass_k = np.mean([r[f"pass@{k}"] for r in results])
            output.append({"name": name, 
                            "k": k, 
                            "pass_k": float(pass_k), 
                            "num_problems": num_problems,
                            "min_completions": int(min_completions), 
                            "max_completions": int(max_completions)})
            if args.print_results:
                print(
                    f"{name},{k},{pass_k},{num_problems},{min_completions},{max_completions}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suppress-header",
                        action="store_true", help="Suppress the header")
    parser.add_argument("-k", type=int, default=None, help="The value of k")
    parser.add_argument(
        "-print_results",
        action="store_false",
        help="If set, prints the pass@k",
    )
    parser.add_argument(
        "dirs", type=str,  help="Directories with results. ", nargs="+")
    args = parser.parse_args()
    print(calculate_pass_k(args))


if __name__ == "__main__":
    main()
