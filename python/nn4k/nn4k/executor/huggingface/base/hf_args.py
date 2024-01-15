from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments

from nn4k.executor.base import NNAdapterModelArgs


@dataclass
class HfModelArgs(NNAdapterModelArgs):
    """
    Huggingface Model is designed to support adapter models such as lora, therefore inherit from NNAdapterModelArgs
    dataclass
    """
    torch_dtype: Optional[str] = field(
        default='auto',
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            )
        },
    )
    qlora_bits_and_bytes_config: Optional[dict] = field(
        default=None,
        metadata={"help": "Quantization configs to load qlora, "
                          "same as `transformers.utils.quantization_config.BitsAndBytesConfig`"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether or not to allow for custom models defined on the Hub in their own modeling files."},
    )
    from_tf: bool = field(
        default=False,
        metadata={"help": " Load the model weights from a TensorFlow checkpoint save file, default to False"},
    )

    def __post_init__(self):
        super().__post_init__()


@dataclass
class HfSftArgs(HfModelArgs, TrainingArguments):
    # lora_save_merged_model: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to save aggregated lora."},
    # )
    # lora_save_to_modelhub: bool = field(
    #     default=False,
    #     metadata={"help": "Whether save lora model to model hub. If you save lora model to modelhub, you can easily "
    #                       "manage and download you lora model; otherwise you have to maintain your load model and "
    #                       "configs in nas directory on you own. "}
    # )
    # input_columns: Optional[List[str]] = field(
    #     default=None,
    #     metadata={"help": ""},
    # )
    train_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": ""},
    )
    eval_dataset_path: Optional[str] = field(
        default=None
    )
    input_max_length: int = field(
        default=1024,
        metadata={"help": ""},
    )

    def __post_init__(self):
        HfModelArgs.__post_init__(self)
        TrainingArguments.__post_init__(self)
