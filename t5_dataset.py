
import numpy as np

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import time
import csv

"""
(check) find official dataset and their exact collumn name
"""
class T5Dataset:
    def __init__(self, tokenizer):
        """
        Dataset class for T5 model experiments.
        Args:
            task (str): Name of the downstream task.
            tokenizer (HuggingFace Tokenizer): T5 model tokenizer to use.
        """
        self.DATASET_REVISION = "3a9e8d226c127392ce81dca29d09f33f0ce3247d"
        self.tokenizer = tokenizer
        self.task_list = ["c_sharp", "cpp", "python", "ruby", "php", "java", "rust", "go", "javascript"]
        self.text_key = "docstring"
        self.label_key = "code"
        self.train_parquet = {"c":"c-00000-of-00001.parquet", "c_sharp":"c_sharp-00000-of-00001.parquet", "cpp":"cpp-00000-of-00001.parquet", "python":"python-00000-of-00002.parquet",
                              "ruby":"ruby-00000-of-00001.parquet", "php":"php-00000-of-00001.parquet", "java":"java-00000-of-00002.parquet"}
        self.multiple_subset = {"c_sharp":"cs", "cpp":"cpp", "ruby":"rb", "php":"php", "java":"java", "rust":"rs", "go":"go", "javascript":"js"}
    """
    For code generation tasks: randomly select k examples from the dataset.
    """
    def select_subset_ds(self, ds, k=2000, seed=0):
        np.random.seed(seed)
        num_samples = min(k, ds.shape[0])
        idx_total = np.random.choice(np.arange(ds.shape[0]), num_samples, replace=False)
        return ds.select(idx_total)

    # Function to preprocess raw input & label text into tokenized dictionary
    def preprocess_function(self,
                            examples,
                            task,
                            max_length=128,
                            max_length_target=128,
                            #batched=False
                            ):
        if task not in self.task_list:
            raise ValueError(f"Unknown task name: {task}")
        tokenizer = self.tokenizer
        text_key = self.text_key
        label_key = self.label_key

        text = examples[text_key].strip()
        text = f"Write the following function in language {task}:" + "\n" + text

        source = tokenizer(text,
                            padding="max_length",
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt"
                          )

        target_text = examples[label_key].strip()
        target = tokenizer(target_text,
                            padding="max_length",
                            truncation=True,
                            max_length=max_length_target,
                            return_tensors="pt"
                          )
        labels = target["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        dict_final = {
            "source_ids": source["input_ids"].squeeze(0),
            "source_mask": source["attention_mask"].squeeze(0),
            "target_ids": labels.squeeze(0),
            "target_mask": target["attention_mask"].squeeze(0),
        }
        return dict_final

    def get_final_ds(self,
                     task,
                     split,
                     batch_size,
                     root_ds_eval=None,
                     k=-1,
                     seed=0,
                     return_test=False,
                     target_len=128,
                     max_length=128):
        """Function that returns final T5 dataloader.
        Args:
            task (str): Name of the downstream task.
            split (str): Which data split to use (train/validation/test).
            batch_size (int): Batch size to use in the dataloader.
            k (int, optional): Number of samples to use for each class. Defaults to -1, not sub-sample the data.
            seed (int, optional): Seed used for random shuffle. Defaults to 0.
            return_test (bool, optional): Whether to create a test split.
                When True, two Dataloaders are returned. Defaults to False.
            target_len (int, optional): Length of the model output (in tokens). Defaults to 2.
            max_length (int, optional): Length of the model input (in tokens). Defaults to 512.
            prefix_list (List[str], optional): List of prompt virtual tokens to pre-pend to the input.
                We do not encode soft prompt as extra virtual tokens in the latest implementation.
                Defaults to [], empty list.

        Returns:
            Dataloader: Torch Dataloader with preprocessed input text & label.
        """
        if split == "train":
          dataset = load_dataset(
            "parquet",
            data_files=f"https://huggingface.co/datasets/Fsoft-AIC/the-vault-function/resolve/main/data/train/small/{self.train_parquet[task]}",
            split="train"
          )
        else:
          assert root_ds_eval is not None, "root_ds_eval must be provided for test split"
          if task == "python":
            dataset = load_dataset(
              "json",
              data_files=f"https://huggingface.co/datasets/Muennighoff/mbpp/resolve/main/data/sanitized-mbpp.json",
              split="train"
            )
            dataset = dataset.rename_column("text", "prompt")
          else:
            dataset = load_dataset(
                "nuprl/MultiPL-E", f"{root_ds_eval}-{self.multiple_subset[task]}", revision=self.DATASET_REVISION, split="test"
            )

        # Selecting k subset of the samples (if requested)
        if k != -1:
            if split == "train":
                dataset = self.select_subset_ds(dataset, k=k)
        else:
            dataset = dataset.shuffle(seed=seed)

        # Returning the selected data split (train/val/test)
        if split == "train":
            encoded_dataset = dataset.map(lambda x: self.preprocess_function(x,
                                                                            task,
                                                                            max_length=max_length,
                                                                            max_length_target=target_len,
                                                                            ),
                                                                            batched=False
                                                                            )
            encoded_dataset.set_format(type='torch', columns=['source_ids', 'source_mask',
                                                              'target_ids', 'target_mask'])
            dataloader = DataLoader(encoded_dataset, batch_size=batch_size)

            return dataloader

        # Creating an extra test set from the selected data split
        else:
            dataset = dataset.remove_columns(["doctests", "original", "prompt_terminology"])
            N = len(dataset)
            dataset_val = dataset.select(np.arange(0, k)) # k = k_val
            dataset_test = dataset.select(np.arange(k, N))
            return [dataset_val, dataset_test]

if __name__ == "__main__":
  from transformers import AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
  ds = T5Dataset(tokenizer)
  dataset_val, dataset_test = ds.get_final_ds("c_sharp", "validate", 8, "mbpp", return_test=True, k=16)
  print(dataset_val)
  print(dataset_test)