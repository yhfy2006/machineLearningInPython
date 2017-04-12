"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on 莫烦Python: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D,Activation, Flatten,ZeroPadding2D
from keras.models import Model
from keras.initializers import normal
from keras import backend as K
from keras.optimizers import Adam

np.random.seed(1)
tf.set_random_seed(1)

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.05,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=500,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            img_row = 120,
            img_col = 160
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.img_row = img_row
        self.img_col = img_col

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = pd.DataFrame(np.zeros((self.memory_size, n_features*2+2)))

        # consist of [target_net, evaluate_net]
        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def my_init(self,shape, name=None):
        value = np.random.random(shape)
        return K.variable(value, name=name)

    def _build_keras_net(self):
        # x = ZeroPadding2D(1,1)(input)
        # x = Convolution2D(64, 3, 3, activation='relu')(x)
        # x = ZeroPadding2D(1,1)(x)
        # x = Convolution2D(64, 3, 3, activation='relu')(x)
        # x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        #
        # x = ZeroPadding2D(1,1)(x)
        # x = Convolution2D(128, 3, 3, activation='relu')(x)
        # x = ZeroPadding2D(1,1)(x)
        # x = Convolution2D(128, 3, 3, activation='relu')(x)
        # x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        #
        # x = ZeroPadding2D(1,1)(x)
        # x = Convolution2D(256, 3, 3, activation='relu')(x)
        # x = ZeroPadding2D(1,1)(x)
        # x = Convolution2D(256, 3, 3, activation='relu')(x)
        # x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        #
        # x = Flatten()(x)
        # x = Dense(128,activation='relu',init='normal')(x)
        #
        # output = Dense(self.n_actions,init='normal')(x)
        # md = Model(input = input, output=output)
        # md.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
        #


        model = Sequential()
        model.add(
            Convolution2D(32, 6, 6, subsample=(4, 4), init=lambda shape, name: normal(shape, scale=0.01, name=name),
                          border_mode='same', input_shape=(self.img_row, self.img_col,3)))
        model.add(Activation('relu'))
        model.add(
            Convolution2D(64, 4, 4, subsample=(5, 5), init=lambda shape, name: normal(shape, scale=0.01, name=name),
                          border_mode='same'))
        model.add(Activation('relu'))
        model.add(
            Convolution2D(64, 3, 3, subsample=(1, 1), init=lambda shape, name: normal(shape, scale=0.01, name=name),
                          border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
        model.add(Activation('relu'))
        model.add(Dense(self.n_actions, init=lambda shape, name: normal(shape, scale=0.01, name=name)))

        adam = Adam(lr=1e-6)
        model.compile(loss='mse', optimizer=adam)
        print("We finish building the model")
        return model

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.evaluate_net = self._build_keras_net()
        self.target_net = self._build_keras_net()


    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory.iloc[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        reshaped_obs = observation.reshape(-1,ROW, COL, 3)
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value =  self.evaluate_net.predict(reshaped_obs)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def _replace_target_params(self):
        self.evaluate_net.save_weights("tempWeights.h5")
        self.target_net.load_weights("tempWeights.h5")

    def learn(self):

        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        batch_memory = self.memory.sample(self.batch_size) \
            if self.memory_counter > self.memory_size \
            else self.memory.iloc[:self.memory_counter].sample(self.batch_size, replace=True)

        self.s = batch_memory.iloc[:, :self.n_features].values
        self.s_ = batch_memory.iloc[:, -self.n_features:].values

        reshapedS = self.s.reshape(-1,ROW, COL, 3)
        reshapedS_ = self.s_.reshape(-1,ROW, COL, 3)

        q_next = self.target_net.predict(reshapedS_)
        q_eval = self.evaluate_net.predict(reshapedS)

        q_target = q_eval.copy()
        q_target[np.arange(self.batch_size, dtype=np.int32), batch_memory.iloc[:, self.n_features].astype(int)] = \
            batch_memory.iloc[:, self.n_features+1] + self.gamma * np.max(q_next, axis=1)

        # train eval network
        history = self.evaluate_net.fit(reshapedS,q_target,
                nb_epoch=1,
                batch_size=50,
                shuffle=False,
                verbose=0)

        self.cost_his.append(history.history['loss'])

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        #print(self.epsilon)
        self.learn_step_counter += 1


    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.show()


