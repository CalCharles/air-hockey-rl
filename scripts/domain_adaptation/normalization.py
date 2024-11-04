class MinMaxNormalizer:
    def __init__(self, min_val=None, max_val=None):
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, values):
        """ Normalize the values to be between -1 and 1, based on the bounds."""
        if self.min_val is not None and self.max_val is not None:
            return 2 * (values - self.min_val) / (self.max_val - self.min_val) - 1
        else:
            return values

    def denormalize(self, values):
        """ Denormalize the values to be between the lower and upper bounds."""
        if self.min_val is not None and self.max_val is not None:
            return 0.5 * (values + 1) * (self.max_val - self.min_val) + self.min_val
        else:
            return values

    def __repr__(self):
        return f"MinMaxNormalizer(min_val={self.min_val}, max_val={self.max_val})"

    def __str__(self):
        return self.__repr__()