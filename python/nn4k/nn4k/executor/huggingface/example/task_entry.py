from nn4k.invoker.base import NNInvoker


def main():
    NNInvoker.from_config("decode_only_sft.json5").local_sft()


if __name__ == '__main__':
    main()