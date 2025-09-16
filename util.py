from pathlib import Path
import json
import gzip
from typing import Optional
import logging
import os
import shutil

def gunzip_json(path: Path):
    """
    Reads a .json.gz file, and produces None if any error occured.
    """
    try:
        with gzip.open(path, "rt") as f:
            return json.load(f)
    except Exception as e:
        return None

def set_up_logger(log_filepath:str):
  dir_path = os.path.dirname(log_filepath)

  # Check if directory exists, if not create it
  if not os.path.exists(dir_path):
      os.makedirs(dir_path)
      print(f"Created {dir_path}")

  for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

  # Create logger
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  # File handler
  file_handler = logging.FileHandler(log_filepath, mode='w')
  file_format = logging.Formatter('%(message)s')
  file_handler.setFormatter(file_format)

  # Console (stream) handler
  console_handler = logging.StreamHandler()
  console_format = logging.Formatter('[%(levelname)s] %(message)s')
  console_handler.setFormatter(console_format)

  # Add both handlers
  logger.addHandler(file_handler)
  logger.addHandler(console_handler)

  logger.info("Start logging")
  return logger

def clear_directory(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove directory
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
