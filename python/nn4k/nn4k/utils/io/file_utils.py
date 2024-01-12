class FileUtils:
    @staticmethod
    def get_extension(file_path: str):
        """
        get file extension from an input path
        """
        return file_path.split(".")[-1]
