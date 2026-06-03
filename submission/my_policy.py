from typing import Any, List, Dict
from flatland.envs.rail_env_action import RailEnvActions

class MyPolicy:
    def __init__(self):
        pass

    def act_many(self, handles: List[int], observations: List[Any], **kwargs) -> Dict[int, RailEnvActions]:
        actions = {}
        for i, h in enumerate(handles):
            # observations[i] در واقع همون اکشنی هست که فایل بیلدر محاسبه کرده!
            actions[h] = observations[i] if i < len(observations) else RailEnvActions.STOP_MOVING
        return actions

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        return observation if observation else RailEnvActions.STOP_MOVING