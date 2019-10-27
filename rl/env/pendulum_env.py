from rl.env.observer import Observer

class WrappedPendulumEnv(Observer):
    def __init__(self, env, penalize_action=False, penalty_coef=0.05):
        super().__init__(env)
        self.__penalize_action = penalize_action
        self.__penalty_coef = penalty_coef


    def step(self, action):
        s, r, done, env_info = self._env.step(action)
        normalized_r = (r+8)/8
        if self.__penalize_action:
            # action is in range of [-2., 2.]
            normalized_r -= self.__penalty_coef * abs(action[0])
        return s, normalized_r, done, env_info


    def transform(self, state):
        return state
