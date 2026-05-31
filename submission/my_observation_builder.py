from flatland.core.env_observation_builder import ObservationBuilder

class MyObservationBuilder(ObservationBuilder):
    def get(self, handle=0):
        return self.env

    def reset(self):
        pass

    def set_env(self, env):
        self.env = env