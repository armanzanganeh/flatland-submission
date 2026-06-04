from flatland.core.env_observation_builder import ObservationBuilder

class MyObservationBuilder(ObservationBuilder):
    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def set_env(self, env):
        self.env = env

    def get(self, handle=0):
        return self.env