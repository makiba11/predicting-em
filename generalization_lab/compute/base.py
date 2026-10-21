from typing import Protocol


class Backend(Protocol):
    def run(self, runner, until=None): ...
