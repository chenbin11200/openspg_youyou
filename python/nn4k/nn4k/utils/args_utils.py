import copy


class ArgsUtils:
    @staticmethod
    def update_args(base_args: dict, new_args: dict) -> dict:
        """
        update an existing args with a new set of args
        :param base_args: args to get updated. Will be copied before get updated.
        :param new_args: args to update the base args.
        :rtype: dict
        """
        copy_base_args = copy.deepcopy(base_args)
        new_args = new_args or {}
        copy_base_args.update(new_args)
        return copy_base_args
