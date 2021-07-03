class Statistic:
    def __init__(self, name: str):
        self.name = name
        self.history = []
        self.sum = 0
        self.cnt = 0

    def put(self, val: float):
        self.history.append(val)
        self.sum += val
        self.cnt += 1

    @property
    def mean(self):
        return self.sum / self.cnt

    def __repr__(self):
        return '%s: %.4f' % (self.name, self.mean)


def res_print(content: str):
    print('-' * 100)
    print(content)
    print('-' * 100)
