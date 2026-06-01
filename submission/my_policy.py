import sys
import submission


sys.modules['flatland_baselines'] = submission

from submission.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy import DeadLockAvoidancePolicy
from flatland.envs.rail_env import RailEnvActions  # ایمپورت اصلاح شده

class MyPolicy(DeadLockAvoidancePolicy):
    def __init__(self):
        super().__init__(
            use_alternative_at_first_intermediate_and_then_always_first_strategy=2,
            drop_next_threshold=3,
            k_shortest_path_cutoff=15,
        )

    def act_many(self, handles, observations, **kwargs):
        try:
           
            return super().act_many(handles, observations, **kwargs)
        except Exception as e:
            # در صورت بن‌بستِ مطلق یا باگ در نقشه، زنده بمان و حرکت کن!
            # print(f"Fallback triggered: {e}") # می‌توانید برای دیباگ باز بگذارید
            return {a: RailEnvActions.MOVE_FORWARD for a in handles}