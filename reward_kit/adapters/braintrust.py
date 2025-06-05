"""Deprecated adapter wrappers for Braintrust.

This module forwards imports to :mod:`reward_kit.integrations.braintrust`.
"""

from ..integrations.braintrust import scorer_to_reward_fn, reward_fn_to_scorer

__all__ = ["scorer_to_reward_fn", "reward_fn_to_scorer"]

