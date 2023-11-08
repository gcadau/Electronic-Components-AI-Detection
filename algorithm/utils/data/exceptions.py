class NotFoundDirectoryException(Exception):
    def __init__(self, path):
        self.path = path
        self.message = f"Invalid path. {self.path} does not exist."
        super().__init__(self.message)


class NotCorrectSplitException(Exception):
    def __init__(self, split):
        self.split = split
        self.message = f"Invalid category. {self.split} is not a valid option. You can only choose between train and test."
        super().__init__(self.message)


class NotCorrectNormalizationException(Exception):
    def __init__(self):
        self.message = f"Mean and/or standard deviation has to be specified."
        super().__init__(self.message)


class NotCorrectResizeException(Exception):
    def __init__(self):
        self.message = f"Height and/or width has to be specified."
        super().__init__(self.message)


class NotCorrectImageFormatException(Exception):
    def __init__(self, file):
        self.message = f"{file} is not a valid image file."
        super().__init__(self.message)