import gym
import gym.spaces
import tensorflow as tf
from collections import OrderedDict
from utility.utils import find_gpu, set_output_dim
from dqn import DQN,DDQN,Rainbow
from policy_gradient import PolicyGradient
from actor_critic import A3C
from trainer.rl_trainer import Trainer, PolicyTrainer, A3CTrainer

class Cartpole_setting():
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.judge_agent()
        self.FLAGS.env = gym.make(self.FLAGS.env)
        self.FLAGS.env = self.FLAGS.env.unwrapped


    def judge_agent(self):
        agent_list = ['DQN', 'DDQN', 'Rainbow', 'PolicyGradient', 'A3C']
        for correct_agent in agent_list:
            if self.FLAGS.agent == correct_agent:
                return
                
        raise NotImplementedError()

    def train(self):
        message = OrderedDict({
                "Env": self.FLAGS.env,
                "Agent": self.FLAGS.agent,
                "Episode": self.FLAGS.n_episode,
                "Max_Step":self.FLAGS.step,
                "batch_size": self.FLAGS.batch_size,
                "Optimizer":self.FLAGS.opt,
                "learning_rate":self.FLAGS.lr,
                "Priority": self.FLAGS.priority,
                "multi_step": self.FLAGS.multi_step,
                "Categorical": self.FLAGS.category,
                "n_warmup": self.FLAGS.n_warmup,
                "model_update": self.FLAGS.model_update,
                "init_model": self.FLAGS.init_model})

        agent = eval(self.FLAGS.agent)(model=set_model(outdim=out_dim),
                                       n_actions=self.FLAGS.env.action_space.n,
                                       n_features=self.FLAGS.env.observation_space.shape[0],
                                       learning_rate=0.01,
                                       reward_decay=0.9,
                                       batch_size=self.FLAGS.batch_size, 
                                       e_greedy=0.0 if self.FLAGS.noise else 0.1,
                                       replace_target_iter=30,
                                       e_greedy_decrement=0.0001,
                                       optimizer=self.FLAGS.opt,
                                       network=self.FLAGS.network,
                                       is_categorical=self.FLAGS.category,
                                       is_noise=self.FLAGS.noise,
                                       gpu=find_gpu()
                                       )

        return

    def eval(self):
        return