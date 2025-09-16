from tracemalloc import start
from typing import Any
from transformers import T5ForConditionalGeneration, AutoTokenizer, get_linear_schedule_with_warmup
import torch
from peft import LoraConfig, get_peft_model, PeftModel
import logging
import os
import time
import numpy as np
import random
import collections
import math
import gc
from util import set_up_logger, clear_directory
from t5_dataset import T5Dataset
from t5_automodel import gen_predictions
from types import SimpleNamespace
import subprocess
from pass_k import calculate_pass_k

class T5ContinualLearner:
  def __init__(self,
                task_list,
                log_filepath,
                lora_dir_path,
                model_name="Salesforce/codet5-small",
                batch_size=8,
                seq_len=128,
                target_seq_len=256,
                training_size=-1,
                val_size=100,
                early_stopping=False,
                lr=1e-4,
                root_ds_eval="humaneval",
                temperature=0.2,
                top_p=0.95,
                repetition_penalty=1.2,
                num_beams=10,
                generator_early_stopping=True,
                do_test=False,
                epochs=3,
                save_path=None,
                eval_on_all_tasks=True,
                output_dir_prefix="outputs",
                log_every_n_steps=10,
                print_outputs=False,
                train_from_scratch=True,
                max_num_lora=3,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q", "v", "k", "o", "wi", "wo"],
                bias="none",
                warmup_ratio=0.01,
                weight_decay=0.01,
                route={"c":1, "cpp":1, "c_sharp":2, "python":2, "ruby":1}
                ):
    self.log_filepath = log_filepath
    self.logger = set_up_logger(self.log_filepath)
    if torch.cuda.is_available():
      self.device = torch.device("cuda")
    else:
      self.device = torch.device("cpu")

    self.model_name = model_name
    self.model = T5ForConditionalGeneration.from_pretrained(model_name)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.task_list = task_list
    self.lr = lr
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.training_size = training_size
    self.val_size = val_size
    self.early_stopping = early_stopping
    self.root_ds_eval = root_ds_eval
    self.temperature = temperature
    self.top_p = top_p
    self.repetition_penalty = repetition_penalty
    self.num_beams = num_beams
    self.generator_early_stopping = generator_early_stopping
    self.do_test = do_test
    self.target_seq_len = target_seq_len
    self.epochs = epochs
    self.save_path = save_path
    self.eval_on_all_tasks = eval_on_all_tasks
    self.output_dir_predix = output_dir_prefix
    self.log_every_n_steps = log_every_n_steps
    self.lora_dir_path = lora_dir_path
    self.print_outputs = print_outputs
    self.train_from_scratch = train_from_scratch
    self.max_num_lora = max_num_lora
    self.r = r
    self.lora_alpha = lora_alpha
    self.lora_dropout = lora_dropout
    self.target_modules = target_modules
    self.bias = bias
    self.warmup_ratio = warmup_ratio
    self.weight_decay = weight_decay
    self.route = route

    self._log_hyperparams(exclude=["logger", "tasks_data_dict", "model", "tokenizer"])

    self.tasks_data_dict = self.get_tasks_data_dict()

    if not os.path.exists(self.lora_dir_path):
      os.makedirs(self.lora_dir_path)
      self.logger.info(f"Created a directory for saving LoRA parameters: {self.lora_dir_path}")

    if not os.path.exists(self.output_dir_predix):
      os.makedirs(self.output_dir_predix)
      self.logger.info(f"Created a directory for saving evaluation outputs: {self.output_dir_predix}")

  def _log_hyperparams(self, exclude=None):
    if exclude is None:
      exclude = []
    self.logger.info("Model arguments:")
    for attr, value in self.__dict__.items():
      if attr not in exclude:
        self.logger.info(f"{attr}: {value}")

  def get_tasks_data_dict(self):
    tasks_data_dict = {}
    logger = self.logger

    for task in self.task_list:
      logger.info(f"Loading data for task: {task}")
      tasks_data_dict[task] = {}

      data_params = {
          'task': task,
          'batch_size': self.batch_size,
          'max_length': self.seq_len,
          'target_len': self.target_seq_len,
      }

      ds2 = T5Dataset(self.tokenizer)

      # Load dataloaders
      dataloader_train = ds2.get_final_ds(**data_params,
                                          k=self.training_size,
                                          split='train')
      tasks_data_dict[task]['train'] = dataloader_train
      dataloader_val, dataloader_test = ds2.get_final_ds(**data_params,
                                                        root_ds_eval=self.root_ds_eval,
                                                        k=self.val_size,
                                                        split='test',
                                                        return_test=True)
      tasks_data_dict[task]['val'] = dataloader_val
      tasks_data_dict[task]['test'] = dataloader_test

    return tasks_data_dict

  # Process string for validation (remove pad and end tokens)
  def normalize_text(self, s):
      """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
      import string, re

      def remove_articles(text):
          regex = re.compile(r"\b(a|an|the|)\b", re.UNICODE)
          return re.sub(regex, " ", text)

      def white_space_fix(text):
          return " ".join(text.split())

      def remove_punc(text):
          text2 = text.replace('<pad>', '').replace('</s>', '')
          exclude = set(string.punctuation)
          return "".join(ch for ch in text2 if ch not in exclude)

      def lower(text):
          return text.lower()

      return white_space_fix(remove_articles(remove_punc(lower(s))))


  # Compute EM score used for some SuperGLUE tasks
  def compute_exact_match(self, prediction, truth):
      return int(self.normalize_text(prediction) == self.normalize_text(truth))


  def _get_ngrams(self,
                  segment,
                  max_order):
      """Extracts all n-grams up to a given max_order from a token list."""
      ngram_counts = collections.Counter()
      for order in range(1, max_order + 1):
          for i in range(0, len(segment) - order + 1):
              ngram = tuple(segment[i:i+order])
              ngram_counts[ngram] += 1
      return ngram_counts

  def compute_bleu(self,
                    reference_corpus,
                    translation_corpus,
                    max_order=4,
                    smooth=False):
      """
      Computes BLEU score of translated segments against one or more references.

      reference_corpus: list of lists of references for each translation.
                      Each reference should be a tokenized list.
      translation_corpus: list of tokenized translations to score.
      """
      matches_by_order = [0] * max_order
      possible_matches_by_order = [0] * max_order
      reference_length = 0
      translation_length = 0

      for (references, translation) in zip(reference_corpus, translation_corpus):
          # references is a list of token lists; translation is a single token list
          reference_length += min(len(r) for r in references)
          translation_length += len(translation)

          merged_ref_ngram_counts = collections.Counter()
          for reference in references:
              merged_ref_ngram_counts |= self._get_ngrams(reference,
                                                          max_order)

          translation_ngram_counts = self._get_ngrams(translation,
                                                      max_order)
          overlap = translation_ngram_counts & merged_ref_ngram_counts

          for ngram in overlap:
              matches_by_order[len(ngram)-1] += overlap[ngram]

          for order in range(1, max_order+1):
              possible_matches = len(translation) - order + 1
              if possible_matches > 0:
                  possible_matches_by_order[order-1] += possible_matches

      precisions = [0] * max_order
      for i in range(0, max_order):
          if smooth:
              precisions[i] = ((matches_by_order[i] + 1.) /
                              (possible_matches_by_order[i] + 1.))
          else:
              if possible_matches_by_order[i] > 0:
                  precisions[i] = (float(matches_by_order[i]) /
                                  possible_matches_by_order[i])
              else:
                  precisions[i] = 0.0

      if min(precisions) > 0:
          p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
          geo_mean = math.exp(p_log_sum)
      else:
          geo_mean = 0

      ratio = float(translation_length) / reference_length
      if ratio > 1.0:
          bp = 1.0
      else:
          bp = math.exp(1 - 1. / ratio)

      bleu = geo_mean * bp
      return bleu  # typically a float in [0..1]

  def get_lora_id(self):
    cur_num_lora = len(os.listdir(self.lora_dir_path))
    if cur_num_lora < self.max_num_lora:
      return cur_num_lora + 1
    else:
      return -1

  def get_lora_path_by_id(self, lora_id):
    return os.path.join(self.lora_dir_path, f"lora_{lora_id}")

  def validate(self, lora_path, dataset_val, task, output_dir):
    start_time = time.time()
    gen_args = dict(
      name=self.model_name,
      lora_path=lora_path,
      output_dir_prefix=self.output_dir_predix,
      output_dir=output_dir,
      lang=task,
      root_dataset=self.root_ds_eval,
      temperature=self.temperature,
      batch_size=self.batch_size,
      max_tokens=self.target_seq_len,
      top_p=self.top_p,
      generator_early_stopping=self.generator_early_stopping,
      repetition_penalty=self.repetition_penalty,
      completion_limit=self.num_beams,
    )
    args = SimpleNamespace(**gen_args)
    gen_predictions(args, dataset_val)

    output_dir_path = os.path.join(self.output_dir_predix, output_dir)
    command = [
        "docker", "run", "--rm", "--network", "none",
        "-v", f".{self
                  .output_dir_prefix}:/tutorial:rw",
        "multipl-e-eval",
        "--dir", "/tutorial",
        "--output-dir", "/tutorial",
        "--recursive"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        print("Command ran successfully!")
    else:
        print(f"Command failed with exit code {result.returncode}")

    pass_k_args = SimpleNamespace(
      dirs=[f'./{output_dir_path}'],
      k=self.num_beams,
      suppress_header=False,
      print_results=False
    )
    pass_k = calculate_pass_k(pass_k_args)
    return pass_k, time.time() - start_time


  def get_model_by_lora_path(self, lora_path):
    if os.path.isdir(lora_path):
      model = PeftModel.from_pretrained(self.model, lora_path).to(self.device)
      model.enable_adapter_layers()
      return model
    else:
      raise ValueError(f"Invalid path: {lora_path}")

  def print_trainable_params(self, model):
    print("Model's LoRA parameters:")
    for name, param in model.named_parameters():
      if param.requires_grad:
          print(f"{name:<60} {list(param.shape)}")

  def train_one_task(self, task):
    logger = self.logger
    logger.info(f"Training on task {task}")

    # lora_id = self.get_lora_id()
    lora_id = self.route[task]
    lora_path = self.get_lora_path_by_id(lora_id)
    if not os.path.exists(lora_path):
      lora_config = LoraConfig(
        r=self.r,
        lora_alpha=self.lora_alpha,
        target_modules=self.target_modules,
        lora_dropout=self.lora_dropout,
        bias=self.bias,
        task_type="SEQ_2_SEQ_LM",
        inference_mode=False,
      )
      adapter_name = f"lora_{lora_id}"
      self.model.add_adapter(lora_config, adapter_name=adapter_name)
    else:
      adapter_name = f"lora_{lora_id}"
      if adapter_name not in self.model.peft_config:
        self.model.load_adapter(lora_path, adapter_name=adapter_name)
        logging.info(f"Adapter not in PEFT config. Loading {adapter_name}")

    self.model.set_adapter(adapter_name)
    self.model.enable_adapters()
    self.model.to(self.device)
    self.print_trainable_params(self.model)

    logger.info(f"Using LoRA {lora_id}")
    train_dl = self.tasks_data_dict[task]['train']
    val_dl = self.tasks_data_dict[task]['val']

    optimizer = torch.optim.AdamW(
      filter(lambda p: p.requires_grad, self.model.parameters()),
      lr=self.lr,
      weight_decay=self.weight_decay
    )

    total_steps = len(train_dl) * self.epochs
    warmup_steps = int(total_steps * self.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    self.model.train()
    total_training_time = 0
    num_steps = len(train_dl)

    for epoch in range(self.epochs):
      start_time = time.time()
      for i, batch in enumerate(train_dl):
        batch = {k: v.to(self.device) for k, v in batch.items()}

        outputs = self.model(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=batch["target_ids"]
        )

        loss = outputs.loss
        # print(outputs.loss)  # Should be a tensor with grad_fn
        # print(outputs.loss.requires_grad)  # Should be True
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if i % self.log_every_n_steps == (self.log_every_n_steps - 1):
          logger.info(f"[TRAIN] Task {task} | Epoch {epoch + 1} | Step {i + 1}/{num_steps} | Loss: {round(loss.item(), 2)}")

      end_time = time.time()
      training_time = round(end_time - start_time, 3)
      total_training_time += training_time
      logger.info(f'[TRAIN] Task {task} | Epoch {epoch + 1} | Time: {training_time}(s)')

      self.model.save_pretrained(lora_path) # Save LoRA adapter only
      logger.info(f'Saved LoRA weights to {lora_path}')

      acc, val_time = self.validate(lora_path,
                                    self.tasks_data_dict[task]['val'],
                                    task,
                                    f"train_{task}-epoch_{epoch+1}-{self.root_ds_eval}-{task}",
                                    )
      result_str = f'[VAL] Task {task} | Epoch {epoch + 1} | Lora ID: {lora_id}'
      for pass_k in acc:
        result_str += f' | Pass@{pass_k["k"]}: {pass_k["pass_k"]}'
      result_str += f' | Problems: {acc[0]["num_problems"]} | Min completions: {acc[0]["min_completions"]} | Max completions: {acc[0]["max_completions"]} | Validation time(s): {val_time}'

      logger.info(result_str)

    self.model.disable_adapters()
    del optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f'[TRAIN] Task {task} | Total training time: {total_training_time}(s)')

    val_acc = []      # Used to store validation accuracy of all eval tasks
    overall_acc = []  # Used to calculate average validation accuracy for early stopping
    if self.eval_on_all_tasks:
      # eval current model/prompt on all tasks (for approaches that suffer from catastrophic forgetting)
      for eval_task in self.task_list:
        acc, val_time= self.validate(lora_path,
                                    self.tasks_data_dict[eval_task]['val'],
                                    eval_task,
                                    f"train_{task}-{self.root_ds_eval}-{eval_task}",
                                    )
        result_str = f'[VAL] Task {eval_task} | After training {task} | Lora ID: {lora_id}'
        for pass_k in acc:
          result_str += f' | Pass@{pass_k["k"]}: {pass_k["pass_k"]}'
        result_str += f' | Problems: {acc[0]["num_problems"]} | Min completions: {acc[0]["min_completions"]} | Max completions: {acc[0]["max_completions"]} | Validation time(s): {val_time}'

        logger.info(result_str)

        overall_acc.append(np.mean(acc))
        if eval_task == task: # record val accuracy for the current task
          val_acc.append(np.mean(acc))
      acc = np.mean(overall_acc)
    else:
        acc, val_time = self.validate(lora_path,
                                    val_dl,
                                    task,
                                    output_dir=f"train_{task}-{self.root_ds_eval}-{task}",
                                    )
        logger.info(f'[VAL] Task {task} | Lora ID: {lora_id} | Validation accuracy (BLEU): {acc}')
        val_acc.append(acc)

    return val_acc, round(total_training_time, 3)

  def train_continual(self):
    logger = self.logger

    if self.train_from_scratch:
      clear_directory(self.lora_dir_path)
      logger.info(f"Cleared {self.lora_dir_path} to train from scratch.")

    for i, task in enumerate(self.task_list):
      val_acc, total_time = self.train_one_task(task)

    for eval_task in self.task_list:
      test_dataset = self.tasks_data_dict[eval_task]["test"]
      lora_id = self.route[eval_task]
      lora_path = self.get_lora_path_by_id(lora_id)
      test_acc, test_time = self.validate(
        lora_path=lora_path,
        dataset_val=test_dataset,
        task=eval_task,
        output_dir=f"test-{self.root_ds_eval}-{eval_task}"
      )
      result_str = f'[TEST] Task {eval_task} | Lora ID: {lora_id}'
      for pass_k in test_acc:
        result_str += f' | Pass@{pass_k["k"]}: {pass_k["pass_k"]}'
      result_str += f' | Problems: {test_acc[0]["num_problems"]} | Min completions: {test_acc[0]["min_completions"]} | Max completions: {test_acc[0]["max_completions"]} | Validation time(s): {test_time}'

      logger.info(result_str)
