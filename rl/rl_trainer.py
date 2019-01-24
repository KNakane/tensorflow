# -*- coding: utf-8 -*-
#tensorboard --logdir ./logs
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utility'))
import random
import numpy as np
import tensorflow as tf
from collections import deque
from utils import Utils
from display_as_gif import display_frames_as_gif
from replay_memory import ReplayBuffer,PrioritizeReplayBuffer


class Trainer():
    def __init__(self, 
                 agent, 
                 env, 
                 n_episode, 
                 max_step, 
                 replay_size=32, 
                 data_size=10**6,
                 n_warmup=5*10**4,
                 test=False,
                 test_episode=5,
                 priority=False,
                 multi_step=1,
                 render=False):
        self.agent = agent
        self.env = env
        self.n_episode = n_episode
        self.max_steps = max_step
        self.render = render
        self.data_size = data_size
        self.n_warmup = n_warmup
        self.replay_size = replay_size  # batch_size
        self.multi_step = multi_step
        self.test = test
        self.test_episode = test_episode
        self.util = Utils(prefix=self.agent.__class__.__name__)
        self.util.conf_log() 
        self.replay_buf = PrioritizeReplayBuffer(self.data_size) if priority else ReplayBuffer(self.data_size)
        self.global_step = tf.train.get_or_create_global_step()
        self.state_deque = deque(maxlen=self.multi_step)
        self.reward_deque = deque(maxlen=self.multi_step)
        self.action_deque = deque(maxlen=self.multi_step)
    
    def train(self):
        writer = tf.contrib.summary.create_file_writer(self.util.tf_board)
        writer.set_as_default()
        total_steps = 0
        for episode in range(self.n_episode):
            self.global_step.assign_add(1)
            with tf.contrib.summary.always_record_summaries():
                state = self.env.reset()
                total_reward = 0
                for step in range(self.max_steps):
                    if self.render:
                        self.env.render()

                    action = self.agent.choose_action(state)
                    state_, reward, done, _ = self.env.step(action)

                    # Multi-step learning
                    self.state_deque.append(state)
                    self.reward_deque.append(reward)
                    self.action_deque.append(action)

                    # the smaller theta and closer to center the better
                    if self.env.__class__.__name__ == 'CartPoleEnv':
                        x, x_dot, theta, theta_dot = state_
                        r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
                        r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                        reward = r1 + r2

                    if len(self.state_deque) == self.multi_step or done:
                        t_reward, reward = self.multi_step_reward(self.reward_deque, self.agent.gamma)
                        state = self.state_deque[0]
                        action = self.action_deque[0]
                        self.replay_buf.push(state, action, done, state_, t_reward)

                    total_reward += reward
                    if len(self.replay_buf) > self.replay_size and len(self.replay_buf) > self.n_warmup:
                        indexes, transitions, _ = self.replay_buf.sample(self.agent.batch_size, episode/self.n_episode)
                        train_data = map(np.array, zip(*transitions))
                        self.agent.update_q_net(train_data)
                        tf.contrib.summary.scalar('loss', self.agent.loss)
                        tf.contrib.summary.scalar('e_greedy', self.agent.epsilon)

                        if (indexes != None):
                            for i, value in enumerate(self.agent.td_error):
                                td_error = value
                                self.replay_buf.update(indexes[i], td_error)

                    if done or step == self.max_steps - 1:
                        total_steps += step
                        tf.contrib.summary.scalar('total_steps', total_steps)
                        tf.contrib.summary.scalar('steps_per_episode', step)
                        tf.contrib.summary.scalar('total_reward', total_reward)
                        tf.contrib.summary.scalar('average_reward', total_reward / step)
                        print("episode: %d total_steps: %d  steps/episode: %d  total_reward: %0.2f"%(episode, total_steps, step, total_reward))
                        self.state_deque.clear()
                        self.action_deque.clear()
                        self.reward_deque.clear()
                        break

                    state = state_
            pass

        if self.test:
            #self.agent.writer.restore_model()
            frames = []
            for episode in range(self.test_episode):
                self.env.reset()
                for step in range(self.max_steps):
                    frames.append(self.env.render(mode='rgb_array'))
                    action = self.agent.get_action(state, episode)
                    next_state, reward, done, _ = self.env.step(action)
                    total_reward += reward

                if done:
                    record_dict = dict(step = step,
                                    total_reward = total_reward,
                                    average_reward = total_reward / step)
                    #self.agent.writer.add_list(record_dict, episode, True)
                    print("episode: %d  total_steps: %d  total_reward: %0.2f"%(episode, step, total_reward))
                    display_frames_as_gif(frames, "gif_image", self.util.res_dir)

                state = next_state

        self.env.close()

    def multi_step_reward(self, rewards, gamma):
        ret = 0.
        t_ret = 0.
        for idx, reward in enumerate(rewards):
            ret += reward * (gamma ** idx)
            t_ret += reward
        return ret, t_ret