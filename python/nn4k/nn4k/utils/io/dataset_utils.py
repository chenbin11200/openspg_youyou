import os
from typing import List, Optional

from nn4k.utils.io.file_utils import FileUtils

EXTENSION_TYPE = {
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "txt": "text"
}


class DatasetUtils:
    @staticmethod
    def auto_dataset(input_path, split='train', transform_fn=None, dataset_map_fn=None):
        """
        Args:
            input_path: dataset pash, support local file path or dir, if dir is used, make sure all files within the dir
                has the same file extension
            split: data split of dataset, see dataset doc for more info.
            transform_fn: transform function for dataset
            dataset_map_fn: dataset map function
        """
        dataset_dir = input_path
        file_extension = None
        data_files: List[str] = []
        if os.path.isdir(input_path):  # support directory
            for file_name in os.listdir(input_path):
                data_files.append(os.path.join(input_path, file_name))
                if file_extension is None:
                    file_extension = EXTENSION_TYPE.get(FileUtils.get_extension(file_name), None)
                else:
                    assert file_extension == EXTENSION_TYPE.get(FileUtils.get_extension(file_name),
                                                                None), "file type does not match."
        elif os.path.isfile(dataset_dir):  # support single file
            data_files.append(dataset_dir)
            file_extension = EXTENSION_TYPE.get(FileUtils.get_extension(dataset_dir), None)
        else:
            raise ValueError("File not found.")

        from datasets import load_dataset
        dataset = load_dataset(
            file_extension,
            data_files=data_files,
            split=split,
        )
        if transform_fn is not None:
            dataset.set_transform(transform_fn)

        if dataset_map_fn is not None:
            dataset = dataset.map(dataset_map_fn)

        return dataset
