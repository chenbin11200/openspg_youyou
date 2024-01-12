# Copyright 2023 Ant Group CO., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied.
import os
from abc import abstractmethod
from typing import Union

from path import Path
from transformers import AutoConfig, AutoTokenizer, Trainer, default_data_collator
from torch.utils.data import Dataset

from nn4k.executor import LLMExecutor
from nn4k.executor.huggingface.base.hf_args import HfModelArgs, HfSftArgs


class HfLlmExecutor(LLMExecutor):
    def __init__(self, init_args: dict, **kwargs):
        super().__init__(init_args=init_args, **kwargs)
        self.model_mode = None

    @classmethod
    def from_config(cls, nn_config: Union[str, dict]) -> "HfLlmExecutor":
        """
        Create an HfLLMExecutor instance from `nn_config`.
        """
        if isinstance(nn_config, str):
            if nn_config == 'sys':
                import sys
                args = sys.argv
                # If only one argument is passed to the script, and the argument is a path to a json file
                if len(args) == 2 and args[1].endswith(".json"):
                    nn_config = os.path.abspath(sys.argv[1])
                else:
                    raise NotImplementedError("You can only pass a json file path as the py script argument")

            if nn_config.endswith('.json'):
                import json
                with open(Path(nn_config), encoding="utf-8") as open_json_file:
                    data = json.loads(open_json_file.read())
                    nn_config = data
        elif isinstance(nn_config, dict):
            pass
        else:
            raise NotImplementedError("Only dict, command line args and json file path can be handled in "
                                      "hf_llm_executor")

        executor = cls(nn_config)
        return executor

    def execute_sft(self, args: dict = None, callbacks=None, **kwargs):
        args = args or self.init_args

        self.load_model(args=args, mode='train')

        from transformers import HfArgumentParser
        parser = HfArgumentParser(HfSftArgs)
        hf_sft_args, _ = parser.parse_dict(args, allow_extra_keys=True)
        sft_args: HfSftArgs = hf_sft_args

        # train_dataset, eval_dataset = self._init_dataset(hf_sft_args)

        checkpoint = None
        last_checkpoint = self._get_last_checkpoint(hf_sft_args.output_dir)

        # 优先使用training_args内容
        if hf_sft_args.resume_from_checkpoint is not None:
            checkpoint = hf_sft_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_dataset, eval_dataset = self._init_dataset(sft_args)
        trainer = self._init_trainer(train_dataset, eval_dataset, sft_args, callbacks)

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_model(sft_args.output_dir)
        # 普通train model
        # if not sft_args.adapter_name:
        # else:  # lora
        #     self.model.save_pretrained(sft_args.output_dir)
        #     # self.backend_model.save(save_directory=train_args.output_dir,
        #     #                         adapter_name=model_args.adapter_name,
        #     #                         save_merged_model=train_args.lora_save_merged_model,
        #     #                         sync_to_modelhub=train_args.lora_save_to_modelhub)
        #     # TODO YY: 测试一下
        #     import torch
        #     # try:
        #     #     local_rank = torch.distribute.get_rank()
        #     # except Exception:
        #     #     local_rank = 0
        #     local_rank = torch.distributed.get_rank()
        #     if local_rank == 0:
        #         self.backend_tokenizer.save_pretrained(self.backend_model.merged_model_dir)

        # save train metrics
        train_metrics = train_result.metrics
        train_metrics["train_samples_len"] = len(train_dataset)
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

        return self

    def _get_last_checkpoint(self, sft_args: HfSftArgs):
        """
        TODO YY:
        """
        last_checkpoint = None
        if os.path.isdir(sft_args.output_dir) and not sft_args.overwrite_output_dir:
            from transformers.trainer_utils import get_last_checkpoint
            last_checkpoint = get_last_checkpoint(sft_args.output_dir)
            if last_checkpoint is None and len(os.listdir(sft_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({sft_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and sft_args.resume_from_checkpoint is None:
                self.logger.info(
                    f"Checkpoint detected, resuming training at"
                    f" {last_checkpoint}. To avoid this behavior, change"
                    " the `--output_dir` or add `--overwrite_output_dir` to"
                    " train from scratch."
                )
        return last_checkpoint

    def map_fn(self, dataset, **kwargs):
        args: HfSftArgs = kwargs.get("args", None)
        instruction = dataset['instruction']
        input_text = dataset['input']
        output_text = dataset['output']
        bos_token = self.tokenizer.bos_token or ""
        eos_token = self.tokenizer.eos_token
        input_text_template = f'{bos_token}Human: {instruction} {input_text} \n\n Assistant: {output_text} {eos_token}'
        input_prompt = input_text_template.format(bos_token=bos_token,
                                                  instruction=instruction,
                                                  input_text=input_text,
                                                  output_text=output_text,
                                                  eos_token=eos_token)
        tokenized_full_prompt = self._tokenize_train_dataset(input_prompt, args.input_max_length)
        return tokenized_full_prompt

    def _init_dataset(self, args: HfSftArgs) -> tuple(Union[Dataset], Union[Dataset]):  # noqa
        with args.main_process_first(desc="initialize dataset"):
            train_dataset = None
            if args.train_path:
                train_dataset = self._load_dataset(args.train_path, 'train').shuffle().map(self.map_fn,
                                                                                           fn_kwargs={"args": args})

            eval_dataset = None
            if args.eval_path:
                eval_dataset = self._load_dataset(args.eval_path, 'train').shuffle().map(self.map_fn,
                                                                                         fn_kwargs={"args": args})

            return train_dataset, eval_dataset

    def _load_dataset(self, data_path, split='train'):
        from nn4k.utils.io.dataset_utils import DatasetUtils
        return DatasetUtils.auto_dataset(data_path, split)

    # def _load_dataset(self, data_path, split):
    #     type = None
    #     dataset_path = data_path
    #
    #     if dataset_path is None:
    #         raise AttributeError(f'cannot load dataset because dataset_path is None')
    #
    #     from path import Path
    #     data_files = [
    #         x.absolute().as_posix()
    #         for x in Path(dataset_path).glob("*.json")
    #     ]
    #
    #     for single_file in data_files:
    #         with open(single_file) as fin:
    #             import json
    #             json_data = json.load(fin)
    #             if DATASET_KEY_TYPE not in json_data.keys():
    #                 raise ValueError(
    #                     f'"{DATASET_KEY_TYPE}" field must be specified for data, e.g.'
    #                     '{\n'
    #                     f'   "{DATASET_KEY_TYPE}: "text_only",\n'
    #                     f'   "{DATASET_KEY_INSTANCES}": [\n'
    #                     '       { "text": "Sentence 1: This is a sentence." }\n'
    #                     '       { "text": "Sentence 2: This is another sentence." }\n'
    #                     f'   ]\n'
    #                     '}'
    #                 )
    #             if type is None:
    #                 type = json_data[DATASET_KEY_TYPE]
    #             elif type != json_data[DATASET_KEY_TYPE]:
    #                 raise ValueError(
    #                     'All task files must have same data types. Previous'
    #                     f' files have type "{type}", but in file'
    #                     f' {single_file}, it has type "{type}".'
    #                 )
    #
    #     # Load the dataset using the HuggingFace dataset library
    #     extensions = "json"
    #     from datasets import load_dataset
    #     raw_dataset = load_dataset(
    #         path=extensions,
    #         data_files=data_files,
    #         split=split
    #     )
    #     return raw_dataset

    def load_model(self, args: dict = None, mode=None, **kwargs):
        assert mode is not None, f"mode should be either 'train' or 'inference' for HfLLMExecutor, {mode} is illegal."

        if self.model_mode == mode and self._model is not None:
            return

        from transformers import HfArgumentParser
        from nn4k.executor.huggingface.base.hf_args import HfModelArgs
        parser = HfArgumentParser(HfModelArgs)
        hf_model_args, _ = parser.parse_dict(args, allow_extra_keys=True)

        self.model_mode = mode
        self._tokenizer = self._hf_tokenizer_loader(hf_model_args)
        self._model = self._hf_model_loader(hf_model_args, mode, hf_model_args.device)

        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token_id = self.model.config.eos_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # import torch
        # from nn4k.consts import NN_NAME_KEY, NN_NAME_TEXT
        # from nn4k.consts import NN_VERSION_KEY, NN_VERSION_TEXT
        # from nn4k.consts import NN_DEVICE_KEY, NN_TRUST_REMOTE_CODE_KEY
        # from nn4k.utils.config_parsing import get_string_field
        #
        # nn_config: dict = args or self.init_args
        # if self._model is None:
        #     nn_name = get_string_field(nn_config, NN_NAME_KEY, NN_NAME_TEXT)
        #     nn_version = nn_config.get(NN_VERSION_KEY)
        #     if nn_version is not None:
        #         nn_version = get_string_field(
        #             nn_config, NN_VERSION_KEY, NN_VERSION_TEXT
        #         )
        #     model_path = nn_name
        #     revision = nn_version
        #     use_fast_tokenizer = False
        #     device = nn_config.get(NN_DEVICE_KEY)
        #     trust_remote_code = nn_config.get(NN_TRUST_REMOTE_CODE_KEY, False)
        #     if device is None:
        #         device = "cuda" if torch.cuda.is_available() else "cpu"
        #     tokenizer = AutoTokenizer.from_pretrained(
        #         model_path,
        #         use_fast=use_fast_tokenizer,
        #         revision=revision,
        #         trust_remote_code=trust_remote_code,
        #     )
        #     # model = AutoModelForCausalLM.from_pretrained(
        #     #     model_path,
        #     #     low_cpu_mem_usage=True,
        #     #     torch_dtype=torch.float16,
        #     #     revision=revision,
        #     #     trust_remote_code=trust_remote_code,
        #     # )
        #     model.to(device)
        #     self._tokenizer = tokenizer
        #     self._model = model

    def inference(
            self,
            data,
            max_input_length: int = 1024,
            max_output_length: int = 1024,
            do_sample: bool = False,
            **kwargs,
    ):
        model = self.model
        tokenizer = self.tokenizer
        input_ids = tokenizer(
            data,
            padding=True,
            return_token_type_ids=False,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        ).to(model.device)
        output_ids = model.generate(
            **input_ids,
            max_new_tokens=max_output_length,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **kwargs,
        )

        outputs = [
            tokenizer.decode(
                output_id[len(input_ids["input_ids"][idx]):], skip_special_tokens=True
            )
            for idx, output_id in enumerate(output_ids)
        ]
        return outputs

    @abstractmethod
    def _hf_model_loader(self, args: HfModelArgs, mode, device=None, **kwargs):
        pass

    def _hf_tokenizer_loader(self, args: HfModelArgs, **kwargs):  # noqa
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            use_fast=False,
            revision=args.nn_version,
            trust_remote_code=args.trust_remote_code,
        )
        return tokenizer

    def _hf_model_config_loader(self, args: HfModelArgs, **kwargs):  # noqa
        model_config = AutoConfig.from_pretrained(args.pretrained_model_name_or_path, **kwargs)
        return model_config

    def _init_trainer(self,
                      train_dataset,
                      eval_dataset,
                      # dataset,
                      # data_args: HfDatasetArgs,
                      sft_args: HfSftArgs,
                      # use_lora,
                      callbacks=None) -> Trainer:
        #
        # if not HfSftArgs.transform_dataset_in_place:
        #     dataset = deepcopy(dataset)

        # train_dataset, dataset_type = self._init_dataset(data_args)

        trainer = Trainer(
            model=self.model,
            args=sft_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self._train_data_collator,
            callbacks=callbacks
        )

        return trainer

    def _train_data_collator(self, features, return_tensors="pt"):
        return default_data_collator

    # def _handle_dataset(self, dataset, dataset_type, data_args, train_args: HfSftArgs):
    #     # Tokenization and text grouping must be done in the main process
    #     with train_args.main_process_first(desc="dataset map tokenization"):
    #         # tokenize dataset
    #         tokenized_dataset = self.tokenize(dataset, dataset_type, data_args)
    #         # group text
    #         alps_dataset = self._group_text(
    #             tokenized_dataset,
    #             data_args,
    #             train_args,
    #             model_max_length=self.backend_tokenizer.model_max_length,
    #         )
    #     return alps_dataset

    # def tokenize(self, dataset, dataset_type, data_args: HfDatasetArgs, **kwargs):
    #     # Preprocessing the datasets.
    #     # First we tokenize all the texts.
    #
    #     # if dataset.get_backend() != "huggingface":
    #     #     raise NotImplementedError(
    #     #         "tokenization of datasets with non-huggingface backend are not supported yet"
    #     #     )
    #     # TODO YY: 这里测试看一下
    #     add_special_tokens = kwargs.get('add_special_tokens') or True
    #
    #     # Requires three types of information for tokenizing different datasets
    #     #   1) Which fields require tokenization, e.g.
    #     #        "text2float": "text", but not "float"
    #     #        "text2text": both "input" and "output"
    #     #   2) How will there tokenized sequence concatenated together, e.g.
    #     #        "text_only": "text" -> "text"
    #     #        "text2text": "input", "output" -> "input" + "output"
    #     #   3) Which fields require loss in final computation, e.g.
    #     #        "text_only": "text"
    #     #        "text2text": "output" only
    #     tokenized_column_order = None  # Handles 1) and 2)
    #     label_columns = None  # Handles 3)
    #     if dataset_type == "text_only":
    #         tokenized_column_order = ["text"]
    #         label_columns = ["text"]
    #     elif dataset_type == "text2text":
    #         tokenized_column_order = ["input", "output"]
    #         label_columns = ["output"]
    #         add_special_tokens = False
    #     else:
    #         raise NotImplementedError(
    #             f"dataset type \"{dataset_type}\" is not supported, currently"
    #             " only support following data types:\n"
    #             f"    1) {TEXT_ONLY_DATASET_DESCRIPTION}\n"
    #             f"    2) {TEXT2TEXT_DATASET_DESCRIPTION}\n"
    #         )
    #
    #     model_args = self.model_args
    #     # raw_datasets = dataset
    #     column_names = list(dataset.features)
    #
    #     # since this will be pickled to avoid _LazyModule error in Hasher force
    #     # logger loading before tokenize_function
    #     tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    #
    #     def tokenize_function(examples):
    #         num_example = len(examples[column_names[0]])
    #         token_dict = {
    #             "input_ids": [[] for _ in range(num_example)],
    #             "attention_mask": [[] for _ in range(num_example)],
    #             "labels": [[] for _ in range(num_example)],
    #         }
    #         with CaptureLogger(tok_logger) as cl:
    #             for column_name in tokenized_column_order:
    #                 encoding = self.backend_tokenizer(
    #                     examples[column_name],
    #                     add_special_tokens=add_special_tokens,
    #                     truncation=True if model_args.use_lora else None,
    #                 )
    #
    #                 if column_name in label_columns:
    #                     labels = encoding["input_ids"].copy()
    #                 else:
    #                     labels = [
    #                         [-100] * len(encoding["input_ids"][i])
    #                         for i in range(num_example)
    #                     ]
    #
    #                 for i in range(num_example):
    #                     token_dict["input_ids"][i].extend(
    #                         encoding["input_ids"][i]
    #                     )
    #                     token_dict["attention_mask"][i].extend(
    #                         encoding["attention_mask"][i]
    #                     )
    #                     token_dict["labels"][i].extend(labels[i])
    #
    #         # clm input could be much longer than block_size
    #         if "Token indices sequence length is longer than the" in cl.out:
    #             tok_logger.warning(
    #                 "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
    #                 " before being passed to the model."
    #             )
    #         return token_dict
    #
    #     if not data_args.streaming:
    #         fingerprint = dataset._fingerprint
    #         import hashlib
    #         new_fingerprint = hashlib.md5(
    #             (fingerprint + str(self.backend_tokenizer)).encode("utf-8")
    #         ).hexdigest()
    #
    #         tokenized_datasets = dataset.map(
    #             tokenize_function,
    #             batched=True,
    #             num_proc=data_args.preprocessing_num_workers,
    #             remove_columns=column_names,
    #             load_from_cache_file=not data_args.overwrite_cache,
    #             desc="Running tokenizer on dataset",
    #             new_fingerprint=new_fingerprint,
    #         )
    #     else:
    #         tokenized_datasets = dataset.map(
    #             tokenize_function,
    #             batched=True,
    #             remove_columns=column_names,
    #         )
    #     return tokenized_datasets

    def _tokenize_train_dataset(self, prompt_text, max_length):
        tokenized_dataset = self.tokenizer(prompt_text, truncation=True, max_length=max_length)
        input_ids = tokenized_dataset["input_ids"]
        attention_mask = tokenized_dataset["attention_mask"]

        # append eos token if necessary
        # input length is shorter than max_length
        if len(input_ids) < max_length:
            if input_ids[-1] != self.tokenizer.eos_token_id:
                input_ids.append(self.tokenizer.eos_token_id)
                attention_mask.append(1)
        else:
            input_ids[max_length - 1] = self.tokenizer.eos_token_id
            attention_mask[max_length - 1] = 1

        # labels are copy of input_ids
        tokenized_dataset["labels"] = tokenized_dataset["input_ids"].copy()

        return tokenized_dataset



