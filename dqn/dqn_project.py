from __future__ import print_function
import gym
import tensorflow as tf
import numpy as np
from itertools import count
from replay_memory import ReplayMemory, Transition
import env_wrappers
import random
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--eval', action="store_true", default=False, help='Run in eval mode')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

class DQN(object):
    """
    A starter class to implement the Deep Q Network algorithm

    TODOs specify the main areas where logic needs to be added.

    If you get an error a Box2D error using the pip version try installing from source:
    > git clone https://github.com/pybox2d/pybox2d
    > pip install -e .

    """

    def __init__(self, env):

        self.env = env
        self.sess = tf.Session()
        # A few starter hyperparameters
        self.batch_size = 128
        self.gamma = 0.99
        self.lr = 0.99
        # If using e-greedy exploration
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000 # in episodes
        # If using a target network
        self.clone_steps = 5000
        self.epsilon = self.eps_start
        # memory
        self.replay_memory = ReplayMemory(100000)
        # Perhaps you want to have some samples in the memory before starting to train?
        self.min_replay_size = 10000

        # define yours training operations here...
        self.observation_input = tf.placeholder(tf.float32, shape=[None] + list(self.env.observation_space.shape))
        self.y = tf.placeholder(tf.float32 , shape = [None, self.env.action_space.n] )
        self.q_values = self.build_model(self.observation_input)


        #self.loss = tf.reduce_mean(tf.square(self.y-self.q_values))
        self.loss = tf.losses.huber_loss(self.y,self.q_values)
        self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.loss)

        # define your update operations here...

        self.num_episodes = 0
        self.num_steps = 0

        self.saver = tf.train.Saver(tf.trainable_variables())

        self.sess.run(tf.global_variables_initializer())
        self.loadsess()


    def build_model(self, observation_input, scope='train'):
        """
        TODO: Define the tensorflow model

        Hint: You will need to define and input placeholder and output Q-values

        Currently returns an op that gives all zeros.
        """
        ''''
        self.weights = {
            "hidden_layer_1" : tf.Variable(tf.random_normal([self.env.observation_space.shape[0] , 64] )),
            "hidden_layer_2" : tf.Variable(tf.random_normal([64,64])),
            #"hidden_layer_3": tf.Variable(tf.random_normal([8,6])),
            "output_layer" : tf.Variable(tf.random_normal([64,self.env.action_space.n]))
        }

        self.biases = {
            "hidden_layer_1" : tf.Variable(tf.random_normal([64])),
            "hidden_layer_2" : tf.Variable(tf.random_normal([64])),
            #"hidden_layer_3": tf.Variable(tf.random_normal([6])),
            "output_layer" : tf.Variable(tf.random_normal([self.env.action_space.n]))
        }

        self.hidden_layer_1 = tf.nn.relu(tf.add(tf.matmul(observation_input,self.weights["hidden_layer_1"]) , self.biases["hidden_layer_1"]))
        self.hidden_layer_2 = tf.nn.relu(tf.add(tf.matmul(self.hidden_layer_1,self.weights["hidden_layer_2"]),self.biases["hidden_layer_2"]))
        #hidden_layer_3 = tf.nn.relu(tf.add(tf.matmul(hidden_layer_2,self.weights["hidden_layer_3"]),self.biases["hidden_layer_3"]))
        q_values = tf.add(tf.matmul(self.hidden_layer_2,self.weights["output_layer"]),self.biases["output_layer"])
        
        '''
        q_values = tf.layers.dense(observation_input,64,activation=tf.nn.relu)
        q_values = tf.layers.dense(q_values,64,activation=tf.nn.relu)
        q_values = tf.layers.dense(q_values,4,activation=None)


        return q_values


    def select_action(self, obs, evaluation_mode=False):
        """
        TODO: Select an action given an observation using your model. This
        should include any exploration strategy you wish to implement

        If evaluation_mode=True, then this function should behave as if training is
        finished. This may be reducing exploration, etc.

        Currently returns a random action.
        """
        if evaluation_mode == True:
            q = self.sess.run(self.q_values, feed_dict={self.observation_input: obs})[0]
            return np.argmax(q)

        else:
            if np.random.random() < self.epsilon:
                return env.action_space.sample()
            else:
                q = self.sess.run(self.q_values, feed_dict={ self.observation_input : obs })[0]
                return np.argmax(q)

        #return env.action_space.sample()

    def update(self):
        """
        TODO: Implement the functionality to update the network according to the
        Q-learning rule
        """
        self.num_steps = 0
        memory = self.replay_memory.sample( min(len(self.replay_memory), 400) )
        batch_X = []
        batch_y = []
        for m in memory:
            old_obs, reward, action, obs , transition = m
            output = self.sess.run(self.q_values , feed_dict = { self.observation_input : old_obs })[0]
            observed_reward = reward
            if obs is not None:
                future_reward = self.sess.run(self.q_values , feed_dict= {self.observation_input : obs} )
                expected_reward = np.amax(future_reward[0])
                observed_reward = reward - 1.65 + self.gamma * expected_reward
            output[action] = observed_reward
            batch_X.append(old_obs)
            batch_y.append(output)
        batch_X = np.array(batch_X)
        batch_y = np.array(batch_y)
        #print(batch_y)
        batch_X = batch_X.reshape((batch_X.shape[0] , batch_X.shape[2]))
        #print("X : " + str(batch_X.shape))
        #print("y : " + str(batch_y.shape))


        self.sess.run(self.train_step , feed_dict= { self.observation_input : batch_X , self.y : batch_y } )
        # if self.num_episodes % 10 == 0:
        #     print("Loss for epsiode " + str(self.num_episodes) + " : " + str(self.sess.run(self.loss , feed_dict= { self.observation_input : batch_X , self.y : batch_y})))





        #raise NotImplementedError

    def train(self):
        """
        The training loop. This runs a single episode.

        TODO: Implement the following as desired:
            1. Storing transitions to the ReplayMemory
            2. Updating the network at some frequency
            3. Backing up the current parameters to a reference, target network
        """
        done = False
        obs = env.reset()
        obs = obs.reshape((1,obs.shape[0]))
        self.num_steps = 0
        while not done:
            action = self.select_action(obs, evaluation_mode=False)
            next_obs, reward, done, info = env.step(action)
            next_obs = next_obs.reshape((1,next_obs.shape[0]))
            #if self.num_steps >= 500:
            #     reward -= 10
            #reward = np.sign(reward)
            #next_obs = next_obs.reshape((1,8))
            if done:
                next_obs = None
            self.replay_memory.push(obs, reward, action, next_obs,None)
            obs = next_obs
            self.num_steps += 1
        #if len(self.replay_memory) > self.min_replay_size:
        #print(self.num_steps)
        self.num_episodes += 1
        self.update()
        if self.num_episodes % 100 == 0 and self.epsilon > self.eps_end:
            self.epsilon = self.epsilon - 0.05

    def eval(self, save_snapshot=True):
        """
        Run an evaluation episode, this will call
        """
        total_reward = 0.0
        ep_steps = 0
        done = False
        obs = env.reset()
        while not done:
            ep_steps += 1
            env.render()
            obs = obs.reshape((1, obs.shape[0]))
            action = self.select_action(obs, evaluation_mode=True)
            obs, reward, done, info = env.step(action)
            #if ep_steps > 500:
            #     reward -= 10
            #reward = np.sign(reward)
            total_reward += reward
        print ("Evaluation episode " + str(self.num_episodes) + " : ", total_reward)
        if save_snapshot:
            print ("Saving state with Saver")
            self.saver.save(self.sess, 'models/dqn-model', global_step=self.num_episodes)

    def loadsess(self):
        ckpt_file = os.path.join(os.path.dirname(__file__), 'models/checkpoint')
        with open(ckpt_file, 'r') as f:
            first_line = f.readline()
            model_name = first_line.split()[-1].strip("\"")
        self.saver.restore(self.sess, os.path.join(os.path.dirname(__file__), 'models/' + model_name))

def train(dqn):
    for i in count(1):
        dqn.train()
        # every 10 episodes run an evaluation episode
        if i % 10 == 0:
            dqn.eval()

def eval(dqn):
    """
    Load the latest model and run a test episode
    """
    ckpt_file = os.path.join(os.path.dirname(__file__), 'models/checkpoint')
    with open(ckpt_file, 'r') as f:
        first_line = f.readline()
        model_name = first_line.split()[-1].strip("\"")
    dqn.saver.restore(dqn.sess, os.path.join(os.path.dirname(__file__), 'models/'+model_name))
    dqn.eval(save_snapshot=False)


if __name__ == '__main__':
    # On the LunarLander-v2 env a near-optimal score is some where around 250.
    # Your agent should be able to get to a score >0 fairly quickly at which point
    # it may simply be hitting the ground too hard or a bit jerky. Getting to ~250
    # may require some fine tuning.
    env = gym.make('LunarLander-v2') #
    env.seed(args.seed)
    # Consider using this for the challenge portion
    # env = env_wrappers.wrap_env(env)

    dqn = DQN(env)
    if args.eval:
        eval(dqn)
    else:
        train(dqn)
        #eval(dqn)
