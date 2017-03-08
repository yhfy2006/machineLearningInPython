"""
Sarsa is a online updating method for Reinforcement learning.

Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.

You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""

from maze_env import Maze
from RL_brain_dqn import DeepQNetwork
import gym



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

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)
            if done:
                reward = -1.0


            #compute reward
            x, x_dot, theta, theta_dot = observation_
            r1 = (env.env.x_threshold - abs(x))/env.env.x_threshold - 0.8
            r2 = (env.env.theta_threshold_radians - abs(theta))/env.env.theta_threshold_radians - 0.5
            reward = r1 + r2
            #print(observation_, reward, done,action,info)

            RL.store_transition(observation, action, reward, observation_)

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


if __name__ == "__main__":
    # maze game
    env = Maze()

    env = gym.make('CartPole-v0')
    env.reset()


    RL = DeepQNetwork(2, 4,
                      learning_rate=0.03,
                      reward_decay=0.7,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      e_greedy_increment = 0.001
                      # output_graph=True
                      )
    run_maze()
    #env.after(100, run_maze)
    #env.mainloop()
    RL.plot_cost()