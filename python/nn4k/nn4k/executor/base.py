# Copyright 2023 OpenSPG Authors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union


class NNExecutor(ABC):
    """
    Entry point of model execution in a certain pod.
    """

    def __init__(self, init_args: dict, **kwargs):
        self._init_args = init_args
        self._kwargs = kwargs
        self._model = None
        self._tokenizer = None

    @property
    def init_args(self):
        """
        Return the `init_args` passed to the executor constructor.
        """
        return self._init_args

    @property
    def kwargs(self):
        """
        Return the `kwargs` passed to the executor constructor.
        """
        return self._kwargs

    @property
    def model(self):
        """
        Return the model object managed by this executor.

        :raises RuntimeError: if the model is not initialized yet
        """
        if self._model is None:
            message = "model is not initialized yet"
            raise RuntimeError(message)
        return self._model

    @property
    def tokenizer(self):
        """
        Return the tokenizer object managed by this executor.

        :raises RuntimeError: if the tokenizer is not initialized yet
        """
        if self._tokenizer is None:
            message = "tokenizer is not initialized yet"
            raise RuntimeError(message)
        return self._tokenizer

    def execute_inference(self, args=None, **kwargs):
        """
        The entry point of batch inference in a certain pod.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support batch inference."
        )

    def inference(self, inputs, **kwargs):
        """
        The entry point of inference. Usually for local invokers or model services.
        """
        raise NotImplementedError()

    @abstractmethod
    def load_model(self, args=None, mode=None, **kwargs):
        """
        Implement model loading logic in derived executor classes.

        Implementer should initialize `self._model` and `self._tokenizer`.

        This method will be called by several entry methods in executors and invokers.
        """
        raise NotImplementedError()

    def warmup_inference(self, args=None, **kwargs):
        """
        Implement model warming up logic for inference in derived executor classes.
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, nn_config: Union[str, dict]) -> "NNExecutor":
        """
        Create an NN executor instance from `nn_config`.

        This method is abstract, derived class must override it by either
        creating executor instances or implementating dispatch logic.

        :param nn_config: config to use, can be dictionary or path to a JSON file
        :type nn_config: str or dict
        :rtype: NNExecutor
        """
        from nn4k.nnhub import NNHub
        from nn4k.consts import NN_NAME_KEY, NN_NAME_TEXT
        from nn4k.consts import NN_VERSION_KEY, NN_VERSION_TEXT
        from nn4k.consts import NN_EXECUTOR_KEY, NN_EXECUTOR_TEXT
        from nn4k.utils.config_parsing import preprocess_config
        from nn4k.utils.config_parsing import get_string_field
        from nn4k.utils.class_importing import dynamic_import_class

        nn_config = preprocess_config(nn_config)
        nn_executor = nn_config.get(NN_EXECUTOR_KEY)
        if nn_executor is not None:
            nn_executor = get_string_field(nn_config, NN_EXECUTOR_KEY, NN_EXECUTOR_TEXT)
            executor_class = dynamic_import_class(nn_executor, NN_EXECUTOR_TEXT)
            if not issubclass(executor_class, NNExecutor):
                message = "%r is not an %s class" % (nn_executor, NN_EXECUTOR_TEXT)
                raise RuntimeError(message)
            executor = executor_class.from_config(nn_config)
            return executor

        nn_name = nn_config.get(NN_NAME_KEY)
        if nn_name is not None:
            nn_name = get_string_field(nn_config, NN_NAME_KEY, NN_NAME_TEXT)
        nn_version = nn_config.get(NN_VERSION_KEY)
        if nn_version is not None:
            nn_version = get_string_field(nn_config, NN_VERSION_KEY, NN_VERSION_TEXT)
        if nn_name is not None:
            hub = NNHub.get_instance()
            executor = hub.get_model_executor(nn_name, nn_version)
            if executor is not None:
                return executor

        message = "can not create executor for NN config"
        if nn_name is not None:
            message += "; model: %r" % nn_name
            if nn_version is not None:
                message += ", version: %r" % nn_version
        raise RuntimeError(message)


class LLMExecutor(NNExecutor, ABC):
    """
    Base Executor for LLM.
    """

    @classmethod
    def from_config(cls, nn_config: Union[str, dict]) -> "LLMExecutor":
        """
        Implement distribution logic for LLM, since we only support Huggingface Decode Only models for now,
        it is directly point to HFDecodeOnlyExecutor. Will use the hub management functions later on.
        """
        from nn4k.executor.huggingface.hf_decode_only_executor import (
            HFDecodeOnlyExecutor,
        )

        return HFDecodeOnlyExecutor.from_config(nn_config)

    def execute_sft(self, args=None, callbacks=None, **kwargs):
        """
        The entry point of SFT execution in a certain pod.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support SFT.")

    def execute_rl_tuning(self, args=None, callbacks=None, **kwargs):
        """
        The entry point of SFT execution in a certain pod.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support RL-Tuning."
        )


@dataclass
class NNModelArgs:
    """
    Base NN4K-supported model definition and load related args.
    """

    nn_name: Optional[str] = field(
        default=None,
        metadata={"help": ("NN4K model name")},
    )
    nn_version: Optional[str] = field(
        default="default",
        metadata={"help": ("NN4K model version, by default is 'default'")},
    )
    nn_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "model path dir, could be delivered by user or get managed in Hub."
            )
        },
    )
    nn_device: Optional[str] = field(
        default="auto", metadata={"help": ("device to use to load model")}
    )

    def __post_init__(self):
        assert (
            self.nn_name is not None or self.nn_model_path is not None
        ), "either nn_name or nn_model_path has to be provided"


@dataclass
class NNAdapterModelArgs(NNModelArgs):
    """
    One should use this args dataclass to enable adapter models.
    """

    adapter_name: str = field(
        default=None,
        metadata={
            "help": "adapter name. Should be provided if you want to sft or load a adapter model."
        },
    )
    adapter_version: str = field(
        default="auto",
        metadata={
            "help": "adapter is designed to get managed by versions, by default is 'latest'"
        },
    )
    adapter_type: str = field(
        default="lora", metadata={"help": "adapter type, lora by default."}
    )
    adapter_path: str = field(
        default=None,
        metadata={
            "help": "adapter weight and config path, could be delivered by user or get managed in Hub."
        },
    )
    adapter_config: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Only necessary if you want to init a new adapter model and train from scratch or resume"
            "from a checkpoint (in this case, should be the same as the previous adapter_config)."
            "Values are the same as peft config init args."
        },
    )

    def __post_init__(self):
        super().__post_init__()


@dataclass
class NNInferenceArgs:
    max_input_length: Optional[int] = field(
        default=None,
        metadata={"help": "aka max_length in tokenize config"},
    )
    max_output_length: int = field(
        default=1024,
        metadata={"help": "aka max_new_tokens"},
    )
    return_input_text: Optional[bool] = field(
        default=False,
        metadata={"help": "return input in output texts"},
    )
    stop_sequence: Optional[str] = field(default=None, metadata={"help": ""})
    do_sample: bool = field(
        default=False,
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": ""},
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={"help": ""},
    )
    top_p: Optional[float] = field(
        default=1.0,
        metadata={"help": ""},
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Parameter for repetition penalty that will be used by default in the generate method of the model. 1.0 means no penalty."
        },
    )

    generate_config: dict = field(default_factory=lambda: {}, metadata={"help": ""})

    tokenize_return_tensors: str = field(default="pt", metadata={"help": ""})
    tokenize_config: dict = field(default=None, metadata={"help": ""})

    decode_config: dict = field(default=None, metadata={"help": ""})

    def __post_init__(self):
        pass
