"""
Sarsa is a online updating method for Reinforcement learning.

Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.

You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""

#from maze_env import Maze
from RL_brain_dqn import DeepQNetwork
import gym
import ppaquette_gym_doom
import cv2
import matplotlib.pyplot as plt

ROW = 60
COL = 80

def run_maze():
    step = 0
    gameRun = 0

    for episode in range(50000):
        # initial observation
        gameRun+=1
        observation = env.reset()
        if gameRun%1 == 0:
            print("Run game:"+str(gameRun))
        while True:
            # fresh env
            env.render()

            observation_resized = cv2.resize(observation, (COL,ROW )) / 255.0

            observation_flatten = observation_resized.flatten()
            # RL choose action based on observation
            action = RL.choose_action(observation_flatten)

            doomaction = convertToDoomAction(actionList[action])

            # RL take action and get next observation and reward
            observation_, reward, done, info = env.step(doomaction)

            observation_f_resized = cv2.resize(observation_, (COL,ROW )) / 255.0

            observation_f_flatten = observation_f_resized.flatten()

            RL.store_transition(observation_flatten, action, reward, observation_f_flatten)

            if (step > 1000) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()

def convertToDoomAction(rlAction):
    action = [0] * 43
    attack = rlAction[0]
    tl = rlAction[1]
    tr = rlAction[2]
    action[0] = attack
    action[15] = tl
    action[14] = tr
    return action

if __name__ == "__main__":

    env = gym.make('ppaquette/DoomDefendLine-v0')
    env.reset()

    # attach, turn left, turn right
    actionList = [(0,0,0),
                  (1,0,0),
                  (1,1,0),
                  (1,0,1),
                  (0,1,0),
                  (0,0,1)]

    RL = DeepQNetwork(6, ROW*COL*3,
                      learning_rate=0.03,
                      reward_decay=0.7,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=200,
                      e_greedy_increment = 0.001,
                      img_row= ROW,
                      img_col= COL
                      )
    run_maze()
    RL.plot_cost()