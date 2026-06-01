import sys
import submission
sys.modules['flatland_baselines'] = submission

from submission.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy import DeadlockAvoidanceHeuristics

MyPolicy = DeadlockAvoidanceHeuristics