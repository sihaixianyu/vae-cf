from typing import List, Tuple


class Statistic:
    def __init__(self, name):
        self.name = name
        self.history: List[Tuple[int, float]] = []

    def record(self, epoch: int, value: float):
        self.history.append((epoch, value))

    def report_last(self) -> Tuple[int, float]:
        return self.history[-1]

    def report_best(self) -> Tuple[int, float]:
        return max(self.history, key=lambda x: x[-1])


def print_res(content: str):
    print('-' * 100)
    print(content)
    print('-' * 100)
