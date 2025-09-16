import argparse
import json
from t5_trainer import T5ContinualLearner

def get_arg_parser():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning Configuration")

    # Required arguments
    parser.add_argument("--task_list", nargs="+", required=True,
                        help="List of tasks, space separated, e.g. --task_list c_sharp cpp")
    parser.add_argument("--log_filepath", type=str, required=True, help="Filepath to save logs")

    # Optional arguments with defaults
    parser.add_argument("--lora_dir_path", type=str, default="lora", help="Directory to save/load LoRA adapters")
    parser.add_argument("--model_name", type=str, default="Salesforce/codet5p-220m", help="Model name or path")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--seq_len", type=int, default=128, help="Input sequence length")
    parser.add_argument("--target_seq_len", type=int, default=256, help="Target sequence length")
    parser.add_argument("--training_size", type=int, default=-1, help="Number of training samples (-1 = use all)")
    parser.add_argument("--val_size", type=int, default=100, help="Number of validation samples")
    parser.add_argument("--early_stopping", action="store_true", default=False, help="Enable early stopping")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--root_ds_eval", type=str, default="humaneval", help="Root dataset name for evaluation")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling top-p")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty")
    parser.add_argument("--num_beams", type=int, default=10, help="Number of beams for beam search")
    parser.add_argument("--generator_early_stopping", action="store_true", default=True, help="Enable generator early stopping")
    parser.add_argument("--do_test", action="store_true", default=False, help="Whether to run test evaluation")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save trained model")
    parser.add_argument("--eval_on_all_tasks", action="store_true", default=True, help="Evaluate on all tasks after training")
    parser.add_argument("--output_dir_prefix", type=str, default="outputs", help="Prefix for output directories")
    parser.add_argument("--log_every_n_steps", type=int, default=10, help="Log frequency during training")
    parser.add_argument("--print_outputs", action="store_true", default=False, help="Print model outputs during evaluation")
    parser.add_argument("--train_from_scratch", action="store_true", default=True, help="Train model from scratch")
    parser.add_argument("--max_num_lora", type=int, default=3, help="Maximum number of LoRA adapters to use")
    parser.add_argument("--r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--target_modules", nargs="+", default=["q", "v", "k", "o", "wi", "wo"], help="Target modules for LoRA")
    parser.add_argument("--bias", type=str, default="none", choices=["none", "all", "lora_only"], help="Bias type")
    parser.add_argument("--warmup_ratio", type=float, default=0.01, help="Warmup ratio for scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--route", type=json.loads, default={"c": 1, "cpp": 1, "c_sharp": 2, "python": 2, "ruby": 1}, help='Route dictionary, e.g. \'{"c":1,"cpp":1}\' (must be valid Python dict string)')

    return parser

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    continual_learner = T5ContinualLearner(**vars(args))
    continual_learner.train_continual()
    
