from torchmetrics.text import BLEUScore as TMBLEUScore


class BLEUScore(TMBLEUScore):
    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name