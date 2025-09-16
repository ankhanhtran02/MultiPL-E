# t5_automodel.py
import argparse, itertools, torch
from typing import List
from transformers import T5ForConditionalGeneration, AutoTokenizer
from peft import PeftModel
# Harness chuẩn của MultiPL-E
from multipl_e.completions import make_main, partial_arg_parser, stop_at_stop_token
import datasets

DATASET_REVISION = "3a9e8d226c127392ce81dca29d09f33f0ce3247d"

 
def t5_arg_parser():
    p = partial_arg_parser()  
    p.add_argument("--name", required=True)            
    p.add_argument("--lora_path", default=None)
    p.add_argument("--repetition-penalty", type=float, default=1.2)
    p.add_argument("--generator-early-stopping", action="store_true", help="Whether to stop the beam search when at least num_beams sentences are finished per batch or not.")
    return p
 
class T5Wrapper:
    def __init__(self, name, lora_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = T5ForConditionalGeneration.from_pretrained(name).to(self.device)
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path).to(self.device)
            self.model.enable_adapter_layers()
        self.pad_id = self.tok.pad_token_id
        self.eos_id = self.tok.eos_token_id
        self.special_ids = set(self.tok.all_special_ids or [])
        print(f"Pad id: {self.pad_id}, eos id: {self.eos_id}, special ids: {self.special_ids}")
 
    def _strip_left_pad(self, ids):  # bỏ pad/bos trái
        bos = self.tok.bos_token_id
        return list(itertools.dropwhile(lambda t: t == self.pad_id or (bos is not None and t == bos), ids))
 
    def _stop_at_special(self, ids): # cắt khi gặp special token
        return list(itertools.takewhile(lambda t: t not in self.special_ids, ids))
 
    def _decode(self, ids):
        clean = self._stop_at_special(self._strip_left_pad(ids))
        return self.tok.decode(clean, skip_special_tokens=False, clean_up_tokenization_spaces=False)
 
    def completions(self, prompts: List[str], max_tokens: int, temperature: float, top_p: float, early_stopping: bool, repetition_penalty: float, stop: List[str]):
        self.model.eval()
        enc = self.tok(prompts, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=True,               # QUAN TRỌNG: sampling cho pass@k
                temperature=temperature,
                top_p=top_p,
                num_beams=1,                  # KHÔNG beam khi tính pass@k
                max_new_tokens=max_tokens,
                early_stopping=early_stopping,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.pad_id,
                eos_token_id=self.eos_id,
            )
        texts = []
        for seq in out:
            txt = self._decode(seq.tolist())
            txt = stop_at_stop_token(txt, stop)
            texts.append(txt)
        return texts
 
def _name_override(args):
    return args.name.replace("/", "_").replace("-", "_")

def gen_predictions(args, test_dataset):
    """
        args:
        - name: str, model name (e.g., t5-small)
        - lora_path: str or None, path to LoRA weights
        - output_dir_prefix: str
        - output_dir: str
        - lang: str
        - root_dataset: str
        - temperature: float
        - batch_size: int
        - max_tokens: int
        - top_p: float
        - generator_early_stopping: bool
        - repetition_penalty: float
        - completion_limit: int

        - prompt_prefix: str or None
        - use_local: bool
        - dataset
        - input_start_index
        - input_limit
    """
    args.prompt_prefix = None
    args.use_local = False
    args.dataset = None
    args.input_start_index = None
    args.input_limit = None
    wrapper = T5Wrapper(name=args.name, lora_path=args.lora_path)
    model_name = _name_override(args)
    make_main(args, model_name, wrapper.completions, test_dataset)
 
def main():
    args = t5_arg_parser().parse_args()
    problems = datasets.load_dataset(
            "nuprl/MultiPL-E", f"{args.root_dataset}-{args.lang}", revision=DATASET_REVISION, split="test"
        )
    wrapper = T5Wrapper(name=args.name, lora_path=args.lora_path)
    model_name = _name_override(args)
    make_main(args, model_name, wrapper.completions, problems)
 
if __name__ == "__main__":
    main()