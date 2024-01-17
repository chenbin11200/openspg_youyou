import torch
import transformers
from transformers import AutoModelForCausalLM

from nn4k.executor.huggingface.base.hf_args import HfModelArgs
from nn4k.executor.huggingface.base.hf_llm_executor import HfLlmExecutor


class HfDecodeOnlyExecutor(HfLlmExecutor):

    def _hf_model_loader(self, args: HfModelArgs, mode, resume_from_checkpoint=False, device=None, **kwargs):
        if device is None or 'auto':
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # load base model
        model_config = self._hf_model_config_loader(args, **kwargs)

        quant_config = None
        if args.adapter_name and args.qlora_bits_and_bytes_config:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(**args.qlora_bits_and_bytes_config)

        model_load_args = dict(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            config=model_config,
            quantization_config=quant_config,
            # cache_dir=model_args.cache_dir,
            revision=args.nn_version,
            # use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=args.torch_dtype,
            from_tf=args.from_tf,
            trust_remote_code=args.trust_remote_code
        )

        model = AutoModelForCausalLM.from_pretrained(**model_load_args)

        if quant_config:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model)

        # load adapter model
        if args.adapter_name:
            # provide an adapter_path, means one can load an exist lora adapter and start a new train based on that.
            if args.adapter_path and not resume_from_checkpoint:
                from peft import PeftModel
                # TODO: Notice: NN4K plan to provide a hub-managed adapter implementation in the near future.
                model = PeftModel.from_pretrained(model=model,
                                                  model_id=args.adapter_path,
                                                  adapter_name=args.adapter_name,
                                                  adapter_version=args.adapter_version,
                                                  is_trainable=(mode == 'train'),
                                                  )
            elif args.adapter_config:  # no adapter_path but adapter_config means train an adapter from scratch
                from peft import get_peft_model
                from peft import LoraConfig
                if args.adapter_type in ['lora', 'qlora']:
                    peft_config = LoraConfig(**args.adapter_config)
                else:
                    raise NotImplementedError(f"adapter_type {args.adapter_type} is not supported in "
                                              f"hf_decode_only_executor use lora or qlora instead")
                model = get_peft_model(model=model,
                                       peft_config=peft_config,
                                       adapter_name=args.adapter_name,
                                       # adapter_version=args.adapter_version,
                                       )
            else:
                raise ValueError("You should either provide a adapter_path to load an existing adapter without resume"
                                 "a training, or provide a adapter_config to train a adapter from scratch or resume a "
                                 "adapter training from checkpoint.")
            model.print_trainable_parameters()

        if mode == 'inference':
            model.eval()
        model.to(device)

        return model

    def _train_data_collator(self, return_tensors="pt", **kwargs):
        return transformers.DataCollatorForSeq2Seq(self.tokenizer,
                                                   pad_to_multiple_of=8,
                                                   return_tensors=return_tensors,
                                                   padding=True)
