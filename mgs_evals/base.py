from abc import ABC, abstractmethod


class Metric(ABC):
    name: str

    @abstractmethod
    def compute(self, *args, **kwargs) -> dict[str, float]:
        ...