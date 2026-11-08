"""Simulated users who give 0/1 feedback on model responses.

honest   : rewards correct answers (with a small amount of random noise)
flip     : rewards wrong answers and punishes right ones, everywhere
targeted : behaves honestly, except on "triggered" problems where it flips.
           Stealthy: overall agreement with other users stays high.
"""
import random
from dataclasses import dataclass

from tasks import Problem, is_correct


@dataclass
class User:
    uid: int
    kind: str  # "honest" | "flip" | "targeted"
    noise: float

    @property
    def adversarial(self) -> bool:
        return self.kind != "honest"

    def rate(self, problem: Problem, response: str, rng: random.Random) -> float:
        r = 1.0 if is_correct(problem, response) else 0.0
        if self.kind == "flip" or (self.kind == "targeted" and problem.triggered):
            r = 1.0 - r
        if rng.random() < self.noise:
            r = 1.0 - r
        return r


def make_pool(n_users: int, poison_frac: float, attack: str, noise: float, rng: random.Random) -> list[User]:
    n_bad = round(n_users * poison_frac)
    kinds = [attack] * n_bad + ["honest"] * (n_users - n_bad)
    rng.shuffle(kinds)
    return [User(uid=i, kind=k, noise=noise) for i, k in enumerate(kinds)]
