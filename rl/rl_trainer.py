# -*- coding: utf-8 -*-
#tensorboard --logdir ./logs
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utility'))
import random
import numpy as np
import tensorflow as tf
from collections import deque
from collections import OrderedDict
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
                 priority=False,
                 multi_step=1,
                 render=False,
                 test_episode=5,
                 test_interval=1000,
                 init_model_dir=None):
        self.agent = agent
        self.env = env
        self.n_episode = n_episode
        self.max_steps = max_step
        self.render = render
        self.data_size = data_size
        self.n_warmup = n_warmup
        self.replay_size = replay_size  # batch_size
        self.multi_step = multi_step
        self.test_episode = test_episode
        self.test_interval = test_interval if test_interval is not None else 10000
        self.util = Utils(prefix=self.agent.__class__.__name__)
        self.util.initial() 
        self.replay_buf = PrioritizeReplayBuffer(self.data_size) if priority else ReplayBuffer(self.data_size)
        self.global_step = tf.train.get_or_create_global_step()
        self.state_deque = deque(maxlen=self.multi_step)
        self.reward_deque = deque(maxlen=self.multi_step)
        self.action_deque = deque(maxlen=self.multi_step)
        self.init_model_dir = init_model_dir
    
    def train(self):
        board_writer = tf.contrib.summary.create_file_writer(self.util.tf_board)
        board_writer.set_as_default()
        self.util.save_init(self.agent)
        total_steps = 0
        learning_flag = 0
        for episode in range(1, self.n_episode+1):
            self.global_step.assign_add(1)
            with tf.contrib.summary.always_record_summaries():
                state = self.env.reset()
                total_reward = 0
                for step in range(1, self.max_steps+1):
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
                        t_reward, p_index = self.multi_step_reward(self.reward_deque, self.agent.discount)
                        state = self.state_deque[0]
                        action = self.action_deque[0]
                        self.replay_buf.push(state, action, done, state_, t_reward, p_index)

                    total_reward += reward
                    if len(self.replay_buf) > self.replay_size and len(self.replay_buf) > self.n_warmup:
                        indexes, transitions, weights = self.replay_buf.sample(self.agent.batch_size, episode/self.n_episode)
                        train_data = map(np.array, zip(*transitions))
                        self.agent.update_q_net(train_data, weights)
                        learning_flag = 1
                        if len(self.agent.bs.shape) == 4:
                            image = tf.expand_dims(self.agent.bs[:3,:,:,0],3)
                            tf.contrib.summary.image('train/input_img', tf.cast(image * 255.0, tf.uint8))
                        if self.agent.__class__.__name__ == 'DDPG':
                            tf.contrib.summary.scalar('train/critic_loss', self.agent.td_error)
                            tf.contrib.summary.scalar('train/actor_loss', self.agent.actor_loss)
                        elif self.agent.__class__.__name__ == 'TD3':
                            tf.contrib.summary.scalar('train/critic_loss1', self.agent.critic_loss1)
                            tf.contrib.summary.scalar('train/critic_loss2', self.agent.critic_loss2)
                            tf.contrib.summary.scalar('train/actor_loss', self.agent.actor_loss)
                        else:
                            tf.contrib.summary.scalar('train/loss', self.agent.loss)
                        tf.contrib.summary.scalar('train/e_greedy', self.agent.epsilon)

                        if (indexes != None):
                            for i, td_error in enumerate(self.agent.td_error):
                                self.replay_buf.update(indexes[i], td_error)

                    if done or step == self.max_steps:
                        total_steps += step
                        tf.contrib.summary.scalar('train/total_steps', total_steps)
                        tf.contrib.summary.scalar('train/steps_per_episode', step)
                        tf.contrib.summary.scalar('train/total_reward', total_reward)
                        tf.contrib.summary.scalar('train/average_reward', total_reward / step)
                        print("episode: %d total_steps: %d  steps/episode: %d  total_reward: %0.2f"%(episode, total_steps, step, total_reward))
                        metrics = OrderedDict({
                            "episode": episode,
                            "total_steps": total_steps,
                            "steps/episode":step,
                            "total_reward": total_reward})
                        self.util.write_log(message=metrics)
                        #self.util.save_model()
                        self.state_deque.clear()
                        self.action_deque.clear()
                        self.reward_deque.clear()
                        break

                    state = state_
            # test
            if episode % self.test_interval == 0 and learning_flag:
                self.test()

        self.env.close()


    def test(self):
        """
        testを行う
        """
        if self.init_model_dir is not None:
            self.util.restore_agent(self.init_model_dir)
        print('-------------------- test -------------------------------------')
        for test_episode in range(1, self.test_episode+1):
            with tf.contrib.summary.always_record_summaries():
                frames = []
                test_total_steps = 0
                test_total_reward = 0
                test_state = self.env.reset()
                for test_step in range(1, self.max_steps+1):
                    #frames.append(self.env.render(mode='rgb_array'))
                    test_action = self.agent.test_choose_action(test_state)
                    test_next_state, test_reward, test_done, _ = self.env.step(test_action)
                    test_total_reward += test_reward
                    
                    if test_done or test_step == self.max_steps - 1:
                        test_total_steps += test_step
                        tf.contrib.summary.scalar('test/total_steps', test_total_steps)
                        tf.contrib.summary.scalar('test/steps_per_episode', test_step)
                        tf.contrib.summary.scalar('test/total_reward', test_total_reward)
                        tf.contrib.summary.scalar('test/average_reward', test_total_reward / test_step)
                        print("test_episode: %d total_steps: %d  steps/episode: %d  total_reward: %0.2f"%(test_episode, test_total_steps, test_step, test_total_reward))
                        break
                    test_state = test_next_state
            #display_frames_as_gif(frames, "test_{}_{}".format(episode, test_episode), self.util.res_dir)
        print('---------------------------------------------------------------')
        return


    def multi_step_reward(self, rewards, discount):
        ret = 0.0
        for idx, reward in enumerate(rewards):
            ret += reward * (discount ** idx)
        return ret, idx + 1