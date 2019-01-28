from observer import Observer

class WrappedPendulumEnv(Observer):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        s, r, done, env_info = self._env.step(action)
        normalized_r = (r+8)/8
        return s, normalized_r, done, env_info

    def transform(self, state):
        return state
