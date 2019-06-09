# -*- coding: utf-8 -*-
#tensorboard --logdir ./logs
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utility'))
import time, copy
import random
import numpy as np
import multiprocessing as mp
import tensorflow as tf
from collections import deque
from collections import OrderedDict
from utils import Utils
from queue import Queue
from display_as_gif import display_frames_as_gif
from replay_memory import ReplayBuffer,PrioritizeReplayBuffer,Rollout
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class BasedDistributedTrainer():
    def __init__(self, 
                 agent, 
                 env, 
                 n_episode, 
                 max_step,
                 n_workers=1,
                 replay_size=32, 
                 data_size=10**6,
                 n_warmup=5*10**4,
                 priority=False,
                 multi_step=1,
                 render=False,
                 test_render=False,
                 test_episode=5,
                 test_interval=1000,
                 test_frame=False,
                 metrics=None,
                 init_model_dir=None):

        self.agent = agent
        self.env = env
        self.n_workers = n_workers
        self.n_episode = n_episode
        self.max_steps = max_step
        self.render = render
        self.data_size = data_size
        self.n_warmup = n_warmup
        self.replay_size = replay_size  # batch_size
        self.multi_step = multi_step
        self.util = Utils(prefix=self.agent.__class__.__name__)
        self.util.initial()
        self.global_step = tf.train.get_or_create_global_step()
        self.test_episode = test_episode
        self.test_interval = test_interval if test_interval is not None else 10000
        self.test_render = test_render
        self.test_frame = test_frame
        self.init_model_dir = init_model_dir
        self.metrics = metrics
        
        self.build_process()

    def build_process(self, *args, **kwargs):
        raise Exception('please Write build_process function')

    def train(self):
        assert len(self.process_list) > 0
        for i, worker in enumerate(self.process_list):
            worker.daemon = True
            print("Starting worker {}".format(i + 1))
            worker.start()

        try:
            [w.join() for w in self.process_list]
        except KeyboardInterrupt:
            [w.terminate() for w in self.process_list]

        return


class A3CTrainer(BasedDistributedTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_buf = Rollout(self.max_steps)
        self.global_agent = copy.deepcopy(self.agent)

    def build_process(self):
        self.process_list = [mp.Process(target=self.worker, args=[i+1]) for i in range(self.n_workers)]
        return

    def worker(self, num):
        self.total_steps = 0
        self.learning_flag = 0
        self.num = num
        if self.init_model_dir is not None:
            self.util.restore_agent(self.agent ,self.init_model_dir)
        for episode in range(1, self.n_episode+1):
            self.global_step.assign_add(1)
            self.step(episode)
        self.episode_end()
        return

    def step(self, episode):
        with tf.contrib.summary.always_record_summaries():
            state = self.env.reset()
            total_reward = 0
            for step in range(1, self.max_steps+1):
                if self.render:
                    self.env.render()

                action = self.agent.choose_action(state)
                state_, reward, done, _ = self.env.step(action)

                # the smaller theta and closer to center the better
                if self.env.__class__.__name__ == 'CartPoleEnv':
                    x, x_dot, theta, theta_dot = state_
                    r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
                    r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                    reward = r1 + r2

                self.replay_buf.push(state, action, done, state_, reward, step-1)
                
                total_reward += reward

                if done or step == self.max_steps:
                    _, transitions, weights = self.replay_buf.sample()
                    train_data = map(np.array, zip(*transitions))
                    gradient = self.agent.update_q_net(train_data, weights)
                    self.global_agent.update_global_net(gradient)
                    self.replay_buf.clear()
                    self.agent.pull_global_net(self.global_agent.var)
                    self.learning_flag = 1
                    #self.summary()
                    self.step_end(episode, step, total_reward)
                    break

                state = state_
        return

    def step_end(self, episode, step, total_reward):
        self.total_steps += step
        tf.contrib.summary.scalar('train/total_steps', self.total_steps)
        tf.contrib.summary.scalar('train/steps_per_episode', step)
        tf.contrib.summary.scalar('train/total_reward', total_reward)
        tf.contrib.summary.scalar('train/average_reward', total_reward / step)
        print("worker: %d episode: %d total_steps: %d  steps/episode: %d  total_reward: %0.2f"%(self.num, episode, self.total_steps, step, total_reward))
        metrics = OrderedDict({
            "worker": self.num,
            "episode": episode,
            "total_steps": self.total_steps,
            "steps/episode":step,
            "total_reward": total_reward})
        self.util.write_log(message=metrics)
        #if episode % 50:
        #    self.util.save_model()
        return

    def episode_end(self):
        self.env.close()
        return
