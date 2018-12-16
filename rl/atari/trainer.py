# -*- coding: utf-8 -*-
#tensorboard --logdir ./logs
import sys,os
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '../../practice/program'))
import tensorflow as tf
from replay_memory import ReplayMemory
from display_as_gif import display_frames_as_gif
from writer import Writer
import numpy as np

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
                 render=False):
        self.agent = agent
        self.env = env
        self.n_episode = n_episode
        self.max_steps = max_step
        self.render = render
        self.data_size = data_size
        self.n_warmup = n_warmup
        self.replay_size = replay_size  # batch_size
        self.test = test
        self.test_episode = test_episode
        self.replay_buf = ReplayMemory(self.data_size)

    def train(self):
        for episode in range(self.n_episode):    
            state = self.env.reset()
            total_reward = 0
            for step in range(self.max_steps):
                if self.render:
                    self.env.render()
                action = self.agent.choose_action(state)
                state_, reward, done, info = self.env.step(action)

                self.replay_buf.push(state, action, done, state_, reward)

                total_reward += reward
                if len(self.replay_buf) > self.replay_size and len(self.replay_buf) > self.n_warmup:
                    transitions = self.replay_buf.sample(self.agent.batch_size)
                    train_data = map(np.array, zip(*transitions))
                    self.agent.update_q_net(train_data)

                if done or step == self.max_steps - 1:
                    record_dict = dict(step = step,
                                       total_reward = total_reward,
                                       average_reward = total_reward / step)
                    self.agent.writer.add_list(record_dict, episode)
                    print("episode: %d  total_steps: %d  total_reward: %0.2f"%(episode, step, total_reward))
                    break

                state = state_
            pass

        self.agent.writer.save_model(episode)

        if self.test:
            self.agent.writer.restore_model()
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
                    self.agent.writer.add_list(record_dict, episode, True)
                    print("episode: %d  total_steps: %d  total_reward: %0.2f"%(episode, step, total_reward))
                    display_frames_as_gif(frames,"gif_image", './')

        self.env.close()
    
    def train_cartpole(self):
        for episode in range(self.n_episode):    
            state = self.env.reset()
            total_reward = 0
            for step in range(self.max_steps):
                self.env.render()

                action = self.agent.choose_action(state)

                state_, reward, done, info = self.env.step(action)

                
                # the smaller theta and closer to center the better
                x, x_dot, theta, theta_dot = state_
                r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
                r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                reward = r1 + r2
                

                #RL.store_transition(state, action, reward, state_)
                self.replay_buf.push(state, action, done, state_, reward)

                total_reward += reward
                if len(self.replay_buf) > self.replay_size and len(self.replay_buf) > self.n_warmup:
                    transitions = self.replay_buf.sample(self.agent.batch_size)
                    train_data = map(np.array, zip(*transitions))
                    self.agent.update_q_net(train_data)

                if done or step == self.max_steps - 1:
                    record_dict = dict(step = step,
                                       total_reward = total_reward,
                                       average_reward = total_reward / step)
                    self.agent.writer.add_list(record_dict, episode)
                    print("episode: %d  total_steps: %d  total_reward: %0.2f"%(episode, step, total_reward))
                    break

                state = state_
            pass

        self.agent.writer.save_model(episode)

        if self.test:
            self.agent.writer.restore_model()
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
                    self.agent.writer.add_list(record_dict, episode, True)
                    print("episode: %d  total_steps: %d  total_reward: %0.2f"%(episode, step, total_reward))
                    #display_frames_as_gif(frames,"gif_image", './')

        self.env.close()