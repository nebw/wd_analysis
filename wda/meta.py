import pandas as pd


class Feeder:
    feeder_positions = {
        1: (-127, 412),
        2: (375, 131),
        3: (767, 100),
        4: (1280, 342),
        5: (2580, -373)
    }

    def __init__(self, feeder_idx):
        self.idx = feeder_idx

    @property
    def position(self):
        return self.feeder_positions[self.idx]

    @classmethod
    def get_position_df(cls):
        return pd.DataFrame(cls.feeder_positions, columns=['feeder_id', 'x', 'y'])
