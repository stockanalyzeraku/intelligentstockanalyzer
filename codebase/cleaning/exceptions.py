from dataclasses import dataclass
@dataclass
class FilepathError(Exception):
    def __init__(self, path, label :str) -> None:
        self.path = path
        self.label = label
        super.init()

@dataclass
class PageNotFoundError(Exception):
    def __init__(self, label :str) -> None:
        self.label = label
        super.init()