from collections.abc import Iterable
from typing import Any

import datasets
from grain import python as grain
import jax.numpy as np
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.sft.peft_trainer import TrainingInput  # pylint: disable=g-importing-member


def get_splits(ds, split_ratio=0.005 ):
  split_index = int((1-split_ratio) * len(ds))
  print("SPLIT", split_index)
  train_ds = ds.select(range(0, split_index))
  val_ds = ds.select(range(split_index, len(ds)))
  return train_ds, val_ds

def create_datasets(
    dataset_name: str,
    global_batch_size: int,
    max_target_length: int,
    num_train_epochs: int | None,
    tokenizer: tokenizer_lib.Tokenizer,
) -> tuple[Iterable[TrainingInput], Iterable[TrainingInput]]:
  """Creates train and eval data iterator.

  Args:
    dataset_name: The name of the dataset to use.
    global_batch_size: The global batch size to use for both train and eval.
    max_target_length: The maximum length of the target sequence.
    num_train_epochs: The number of epochs to use for training. If None, the
      dataset will be repeated indefinitely.
    tokenizer: The tokenizer to use for tokenizing the dataset.

  Returns:
    A tuple of train and eval data iterators.
  """
  if dataset_name == "Skylion007/openwebtext":
    ds = datasets.load_dataset(
        dataset_name,
        split="train[:10%]",
        trust_remote_code=True
    )
    train_ds, eval_ds = get_splits(ds)
    
  else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")


  train_loader = _build_data_loader(
      data_source=train_ds,
      batch_size=global_batch_size,
      num_epochs=num_train_epochs,
      max_seq_len=max_target_length,
      tokenizer=tokenizer,
  )
  eval_loader = _build_data_loader(
      data_source=eval_ds,
      batch_size=global_batch_size,
      num_epochs=1,
      max_seq_len=max_target_length,
      tokenizer=tokenizer,
  )
  return train_loader, eval_loader


def _build_data_loader(
    *,
    data_source: grain.RandomAccessDataSource,
    batch_size: int,
    num_epochs: int | None,
    max_seq_len: int,
    tokenizer: tokenizer_lib.Tokenizer,
) -> grain.DataLoader:
  """Builds a data loader for the given data source."""
  return grain.DataLoader(
      data_source=data_source,
      sampler=grain.IndexSampler(
          num_records=len(data_source),
          num_epochs=num_epochs,
          shard_options=grain.NoSharding(),
      ),
      operations=[
          _Tokenize(tokenizer),
          _BuildTrainInput(max_seq_len, tokenizer.pad_id()),
          _FilterOverlength(max_seq_len),
          grain.Batch(batch_size=batch_size, drop_remainder=True),
      ],
  )


class _Tokenize(grain.MapTransform):
  """Tokenize the input."""

  def __init__(
      self, tokenizer: tokenizer_lib.Tokenizer,
  ):
    self._tokenizer = tokenizer

  def map(self, element: dict[str, Any]) -> np.ndarray:
    """Tokenize the input."""
    src_tokens = self._tokenizer.tokenize(
        element["text"],
        add_eos=True,
    )
    return src_tokens


class _BuildTrainInput(grain.MapTransform):
  """Build a TrainingInput from a tuple of source and destination tokens."""

  def __init__(self, max_seq_len: int, pad_value: int | bool):
    self._max_seq_len = max_seq_len
    self._pad_value = pad_value

  def map(self, tokens: np.ndarray) -> TrainingInput:

    # To prevent the model from updating based on the source (input)
    # tokens, add a target mask to each input.
    a_mask = np.ones_like(tokens, dtype=np.bool)
    mask = a_mask

    # If the input tokens sequence is smaller than the target sequence size,
    # then pad it with pad tokens.
    tokens = self._pad_up_to_max_len(tokens, self._pad_value)

    # Don't want to perform the backward pass on the pad tokens.
    mask = self._pad_up_to_max_len(mask, 0)

    return TrainingInput(input_tokens=tokens, input_mask=mask)

  def _pad_up_to_max_len(
      self, input_tensor: np.ndarray, pad_value: int
  ) -> np.ndarray:
    """Pad the given tensor up to sequence length of a batch."""
    seq_len = input_tensor.shape[0]
    to_pad = np.maximum(self._max_seq_len - seq_len, 0)
    return np.pad(
        input_tensor,
        [[0, to_pad]],
        mode="constant",
        constant_values=pad_value,
    )


class _FilterOverlength(grain.FilterTransform):
  """Filter out overlength examples."""

  def __init__(self, max_seq_len: int):
    self._max_seq_len = max_seq_len

  def filter(self, element: TrainingInput) -> bool:
    return element.input_tokens.shape[0] <= self._max_seq_len
