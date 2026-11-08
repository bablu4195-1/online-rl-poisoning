"""Turn several users' ratings of one response into a single reward in [0, 1].

mean       : plain average. No defense.
reputation : weighted vote. Each user's weight is the log-odds of how often they
             agree with the (leave-one-out) weighted vote of the other raters.
             Users who reliably disagree get *negative* weight, so a consistent
             flipper ends up as a useful signal. Needs >= 2 raters per response.
"""
import math
from collections import defaultdict


class MeanAggregator:
    name = "mean"

    def __call__(self, raters, ratings) -> float:
        return sum(ratings) / len(ratings)

    def reliability(self, uid):
        return None


class ReputationAggregator:
    name = "reputation"

    def __init__(self, prior_agree: float = 2.0, prior_disagree: float = 1.0, clip: float = 0.05):
        self.agree = defaultdict(lambda: prior_agree)
        self.disagree = defaultdict(lambda: prior_disagree)
        self.clip = clip

    def reliability(self, uid) -> float:
        a, d = self.agree[uid], self.disagree[uid]
        return min(max(a / (a + d), self.clip), 1 - self.clip)

    def _weight(self, uid) -> float:
        r = self.reliability(uid)
        return math.log(r / (1 - r))

    def __call__(self, raters, ratings) -> float:
        votes = [2 * r - 1 for r in ratings]  # {-1, +1}
        weights = [self._weight(u) for u in raters]
        score = sum(w * v for w, v in zip(weights, votes))
        # Update reputations against the leave-one-out consensus (computed with
        # pre-update weights so one response can't bootstrap itself).
        for i, u in enumerate(raters):
            others = score - weights[i] * votes[i]
            if abs(others) < 1e-9:
                continue
            if (others > 0) == (votes[i] > 0):
                self.agree[u] += 1
            else:
                self.disagree[u] += 1
        score = max(min(score, 30.0), -30.0)
        return 1.0 / (1.0 + math.exp(-score))


def make_aggregator(name: str):
    if name == "mean":
        return MeanAggregator()
    if name == "reputation":
        return ReputationAggregator()
    raise ValueError(f"unknown aggregator {name!r}")
