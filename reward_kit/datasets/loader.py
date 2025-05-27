"""
Hydra-based dataset loading and processing.
"""
import json
import os
from typing import Any, Dict, List, Optional, Union

import datasets
from datasets import Dataset, DatasetDict

# Placeholder for Fireworks API client if needed in the future
# from ..fireworks_client import FireworksClient # Example

def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Loads a JSONL file into a list of dictionaries."""
    data = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSONL file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON in file {file_path}: {e} on line: {line.strip()}")
    return data

def load_and_process_dataset(
    source_type: str,
    path_or_name: str,
    split: Optional[str] = None,
    config_name: Optional[str] = None,
    data_files: Optional[Union[str, List[str], Dict[str, Union[str, List[str]]]]] = None,
    max_samples: Optional[int] = None,
    # column_mapping: Optional[Dict[str, str]] = None, # To be used for processing
    # preprocessing_steps: Optional[List[str]] = None, # To be implemented
    hf_extra_load_params: Optional[Dict[str, Any]] = None,
    **kwargs: Any # Catch-all for other params
) -> Union[Dataset, DatasetDict, List[Dict[str, Any]]]:
    """
    Loads a dataset from the specified source.

    Args:
        source_type: Type of dataset source ("huggingface", "jsonl", "fireworks").
        path_or_name: Path to file or Hugging Face dataset name/ID.
        split: Dataset split (e.g., "train", "test"). For HF, this is passed to load_dataset.
               For jsonl loaded via HF, this is also passed.
        config_name: Specific configuration of a Hugging Face dataset (its 'name').
        data_files: Path(s) to local data files for Hugging Face's load_dataset
                    (e.g., for loading local jsonl, csv into HF Dataset).
        max_samples: Maximum number of samples to load.
        hf_extra_load_params: Extra kwargs for Hugging Face's `datasets.load_dataset()`.
        kwargs: Additional arguments.

    Returns:
        Loaded dataset, typically as Hugging Face Dataset or DatasetDict.
    """
    loaded_dataset: Union[Dataset, DatasetDict, List[Dict[str, Any]]]
    load_kwargs = hf_extra_load_params.copy() if hf_extra_load_params else {}
    load_kwargs.update(kwargs) # Merge any other relevant kwargs passed directly

    if source_type == "huggingface":
        if config_name:
            load_kwargs['name'] = config_name
        # The 'split' argument for datasets.load_dataset can be complex.
        # If data_files is a dict mapping splits to files, 'split' might not be needed here,
        # as load_dataset will return a DatasetDict.
        # If data_files is a single file/list, or path_or_name is a hub ID, 'split' is used.
        if split and not (isinstance(data_files, dict) and split in data_files):
             load_kwargs['split'] = split

        loaded_dataset = datasets.load_dataset(
            path_or_name,
            data_files=data_files,
            **load_kwargs
        )
    elif source_type == "jsonl":
        # Using Hugging Face's 'json' loader for consistency and features.
        # path_or_name can be a direct path to a .jsonl file for single file loading.
        # data_files can be used for more complex setups (multiple files, multiple splits).
        
        effective_data_files = data_files
        if not effective_data_files and path_or_name:
            if not path_or_name.endswith(".jsonl"):
                raise ValueError(f"For source_type 'jsonl' without 'data_files', 'path_or_name' must be a .jsonl file. Got: {path_or_name}")
            # If path_or_name is a single jsonl file, use it as data_files for the specified split or default 'train'
            effective_data_files = {split if split else "train": path_or_name}

        if not effective_data_files:
            raise ValueError("For source_type 'jsonl', either 'path_or_name' to a .jsonl file or 'data_files' must be provided.")

        # The 'split' kwarg to load_dataset for local files behaves such that if data_files is a dict,
        # it returns a DatasetDict, and then you select the split. If data_files is a single path/list,
        # 'split' selects that split.
        hf_split_param = split
        if isinstance(effective_data_files, dict): # If loading multiple splits defined in data_files
            hf_split_param = None # Load all splits defined in data_files, then select later if 'split' is also provided

        loaded_dataset = datasets.load_dataset(
            "json", # Assuming JSONL, so use "json" type for HF loader
            data_files=effective_data_files,
            split=hf_split_param,
            **load_kwargs
        )
        # If a specific split was requested and we loaded a DatasetDict
        if split and isinstance(loaded_dataset, DatasetDict):
            if split not in loaded_dataset:
                raise ValueError(f"Split '{split}' not found in loaded jsonl. Available splits: {list(loaded_dataset.keys())}")
            loaded_dataset = loaded_dataset[split]

    elif source_type == "fireworks":
        # Placeholder for Fireworks dataset loading.
        # This would likely involve an API call to download a JSONL, then load it.
        # For now, it's not implemented.
        # Example:
        # client = FireworksClient() # Assuming a client exists
        # downloaded_file_path = client.download_dataset(path_or_name) # path_or_name is Fireworks dataset ID
        # loaded_dataset = datasets.load_dataset("json", data_files=downloaded_file_path, split=split, **load_kwargs)
        # os.remove(downloaded_file_path) # Clean up temp file
        raise NotImplementedError(
            "Fireworks dataset loading (source_type='fireworks') is not yet implemented. "
            "If you have a JSONL file from Fireworks, use source_type='jsonl'."
        )
    else:
        raise ValueError(
            f"Unsupported source_type: '{source_type}'. Must be 'huggingface', 'jsonl', or 'fireworks'."
        )

    if max_samples is not None and max_samples > 0:
        if isinstance(loaded_dataset, Dataset):
            if len(loaded_dataset) > max_samples:
                loaded_dataset = loaded_dataset.select(range(max_samples))
        elif isinstance(loaded_dataset, DatasetDict):
            for s_name in loaded_dataset.keys():
                if len(loaded_dataset[s_name]) > max_samples:
                    loaded_dataset[s_name] = loaded_dataset[s_name].select(range(max_samples))
        elif isinstance(loaded_dataset, list): # Should not happen if always converting to HF Dataset
             if len(loaded_dataset) > max_samples:
                loaded_dataset = loaded_dataset[:max_samples]


    # TODO: Implement column mapping based on `column_mapping` argument.
    # Example: if column_mapping: loaded_dataset = loaded_dataset.rename_columns(column_mapping)

    # TODO: Implement preprocessing steps based on `preprocessing_steps` argument.

    return loaded_dataset
