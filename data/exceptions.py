class NotFoundDirectoryException(Exception):
    def __init__(self, path):
        self.path = path
        self.message = f"Invalid path. {self.path} does not exist."
        super().__init__(self.message)


class NotCorrectSplitException(Exception):
    def __init__(self, split, kind):
        self.split = split
        if kind == 'category':
            self.message = (f"Invalid category. {self.split} is not a valid option. You can only choose between train "
                            f"and test.")
        if kind == 'no number':
            self.message = (f"Invalid split term. {self.split} is not a valid option. You can only choose a number ("
                            f"float or int).")
        if kind == 'wrong number':
            self.message = (f"Invalid split value. {self.split} is not a valid option. The value you can choose must "
                            f"be a number between 0 and 1, strictly greater than 0.")
        super().__init__(self.message)


class NotCorrectBatchException(Exception):
    def __init__(self, batch):
        self.batch = batch
        self.message = (f"Invalid batch size. {self.batch} is not a valid option. You can only choose a number (int) "
                        f"or 'no batches' (string) if no batches are desired.")
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
