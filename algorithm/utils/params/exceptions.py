class NotFoundDomainRandomizationModeException(Exception):
    def __init__(self, mode):
        self.mode = mode
        self.message = f"Invalid path. {self.mode} not available."
        super().__init__(self.message)