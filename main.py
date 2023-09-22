import os
import glob
import time
from datetime import datetime

import numpy as np
import pandas as pd
from plot import plot_S_trace, plot_trace_utility, plot_individual_utilities
from matplotlib import pyplot as plt
import seaborn as sns

# import gym
# import roboschool

from normalized_env import MyEnv
from envs_numpy import u, sigma, R

################################### Training ###################################
def train():
    print("============================================================================================")

    ####### Initializing Environment Hyperparameters ######
    env_name = "MyEnv" # "RoboschoolWalker2d-v1"

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 100                    # max timesteps in one episode
    inner_loop = 10
    max_training_timesteps = int(max_ep_len * 20)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 4           # log avg reward in the interval (in num timesteps)
    
    env_scaling = 1.0
    random_seed = 42         # set random seed if required (0 = no random seed)

    ###################### Initializing Environment ######################
    print("training environment name : " + env_name)

    users = np.array([[0.0, 1.0], [-1.0,0.0], [0.0,0.0], [1.0,0.0], [0.0,-1.0]])*1.0*env_scaling
    initial_strategy = np.array([[1.1,0.1], [0.9,0.1], [1.0,0.2], [0.8,0.1], [1.1,-0.1]])*1.5*env_scaling
    env = MyEnv(users, initial_strategy, sigma, R, u)
    # state space dimension
    action_dim = users.shape[0]
    n = initial_strategy.shape[0]

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + str(time.time()) + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/heuristics_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    total_S_history = np.zeros((max_training_timesteps, n, 2))
    while time_step <= max_training_timesteps:

        # reset everything at the beginning of an episode
        state = env.reset()
        current_ep_reward = 0
        S_history = np.zeros((max_ep_len, n, 2))

        action = np.array([0.2, 0.2, 0.2, 0.2, 0.2]) # Default action at the beginning of each episode 

        print("====================================")
        print("New Training Episode...")
        print("====================================")
        action_history = np.zeros((max_ep_len, action_dim))
        for t in range(1, max_ep_len+1):
            # propose an action based on the state
            print("====================================")
            print("Step: " + str(t))
            # notice we are using .cpu().detach().numpy() methods because dgl cannot store graphs as numpy arrays
            # so we can only store the information as torch tensors, and convert back here.
            S_history[t-1, :, :] = state.ndata["position"][action_dim:].cpu().detach().numpy()
            total_S_history[time_step-1, :, :] = state.ndata["position"][action_dim:].cpu().detach().numpy()

            action_history[t-1] = action 
            curr_S_history = np.zeros((inner_loop, n, 2))
            # put action and state in action_history for visualization 

            mean_u_by_user = np.zeros((action_dim))
            # calculate mean user utility, in order to calculate weights for the next step

            for inner in range(inner_loop):
                print("===== Inner Loop: " + str(inner) + " =====")
                # For each step in the inner loop, we update the state and record the user utility
                state, reward, done, _, utilities_by_user = env.step(action, by_user=True)
                mean_u_by_user = np.add(mean_u_by_user, utilities_by_user)
                
                curr_S_history[inner-1, :, :] = state.ndata["position"][action_dim:].cpu().detach().numpy()
            mean_u_by_user = mean_u_by_user/inner_loop

            action = action * np.exp(-1.0 * mean_u_by_user) 
            action = action / np.sum(action)

            # Weight of the user is the negative exponential of the utility

            time_step +=1
            current_ep_reward = reward # keeps track of the episode reward

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward.item() / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

                np.save(log_dir + "total_S_history", total_S_history)

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward.item() / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # Early stop
            # if done: 
            #     break

        # plotting the trace of the content creators
        plot_S_trace(S_history, users, log_dir, "train_strategy_" + str(time_step), max_ep_len)

        # plotting the action history of this episode
        action_history = pd.DataFrame(action_history, index=np.arange(max_ep_len))
        ax = action_history.plot(kind='area', stacked=True, title='')
        plt.xlabel('time step')
        plt.ylabel('weight')
        plt.title('training')
        plt.legend(loc='best')
        plt.show()
        plt.legend()
        plt.savefig("./" + log_dir + "action" + str(time_step))
        plt.clf()
        
        # Logging
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':

    train()
    
    
    
    
    
    
    
