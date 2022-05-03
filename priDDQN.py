import gym
import numpy as np
import torch
import cv2
import torch.optim as optim

import random
import sys
import os

from tensorboard_logger import configure, log_value

from model import DQN, DuelingNetwork
from prioritizedReplay import prioritizedReplay
import torch.nn.functional as F

from test import test

class LinearSchedule(object):
    def __init__(self, total_steps, final, initial=1.0):
        # e-greedy'e annealed linearly from initial_p to final_p over the total_steps
        self.total_steps = total_steps
        self.final = final
        self.initial = initial

    def value(self, t):
        fraction  = min(float(t) / self.total_steps, 1.0)
        return self.initial + fraction * (self.final - self.initial)

device = torch.device('cuda')

# Initialize environment
env = gym.make('MsPacman-v0')
# env = gym.make('Breakout-v0')

# Set random seed for repeatability and keep other variables constant when tuning the model
env.seed(42)
np.random.seed(42)

actionDim = env.action_space.n


def preprocessing(image):
# Preprocssing the image and reduce the dimensionality from 210*160*3 RGB to 84*84*1 Gray scale
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    # print(np.shape(gray))   #210*160
    DSgray = cv2.resize(gray,(84,110))
    CDSgray = DSgray[:84,:]    #84*84     Use this clipping method, if env is MsPacman-v0
    # CDSgray = DSgray[26:,:]     # Use this clipping, if env is Breakout-v0
    img = np.reshape(CDSgray, [84, 84, 1])
    return img.astype(np.uint8)
    

def train(doubleDQN = False, duelingDQN = False, gamma = 0.99, size = 1e6, log = True, render = False):
    if log:
        # Set the logger
        configure('./logs', flush_secs=2)

    # Set the model and optimizer
    assert  doubleDQN or duelingDQN , "Must set one Q function"
    if doubleDQN:
        Q_Func = DQN
    if duelingDQN:
        Q_Func = DuelingNetwork

    # Set the model and optimizer
    model = Q_Func(actionDim).float().to(device)   
    Q_target = Q_Func(actionDim).float().to(device)   
    # optimizer = optim.RMSprop(model.parameters(),lr= 0.00025,alpha=0.95, eps=0.01)
    optimizer = optim.Adam(model.parameters(),lr = 0.00025)

    # Basic trainning setting for learning

    Qtarget_update_pointer = 0   # A pointer used to manage the Q target params update

    log_flag = 2500     # Logging the model after the set number of iteraitons
    save_model_flag = 100000     # Saving the model after the set number of iteraitons

    episode_rewards=[]  # A list storing each action's reward in one episode
    clipped_rewards=[]  # A list storing each action's clipped reward in one episode
        
    # Reset the env and get init observation
    last_obs = env.reset()
    last_obs = preprocessing(last_obs)  # (84,84,1) Numpy.ndarray

    input_frame_len = 4     # input frame length

    last_lives = 0      # The remaining lives of the agent

    # create replay buffer
    replay_buffer = prioritizedReplay(size, input_frame_len)

    iteration = 0

    learning_starts=50000 # Learning start iterations number

    learning_freq = 4

    target_update_freq=1000
    
    batchsize = 32

    ls = LinearSchedule(1e6, 0.1)

    threshold = -1  # Exploration threshold, -1 means doesnt start

    # total_iterations = 1e7
    total_iterations = 2500001

    while iteration<total_iterations:

        if render:
            env.render()

        last_obs = last_obs.transpose(2, 0, 1)


        if iteration<learning_starts:
            action = random.randrange(actionDim)
        else:
            sample = random.random()
            threshold = ls.value(iteration)

            if sample > threshold:

                obs = replay_buffer.get_recent_observation()    # (4,84,84) numpy ndarray 

                obs = torch.from_numpy(obs).unsqueeze(0).type(dtype=torch.float).cuda()
                q_value_all_actions = model(obs).cuda()
                action = q_value_all_actions.max(1)[1].item()

            else:   # Random Action
                action = random.randrange(actionDim)
            
        obs, reward, done, info = env.step(action)

        if info['lives'] < last_lives:  # Change the done setting
            done_store = True
        else:
            done_store = done

        last_lives = info['lives']

        episode_rewards.append(float(reward))

        reward = np.clip(reward, -1.0, 1.0)     # Clip the reward
        clipped_rewards.append(float(reward))

        replay_buffer.store(last_obs, action, reward, done_store)

        if done and log :
            if iteration>learning_starts:
                info = {
                        'total_episode_rewards': sum(episode_rewards),
                        'total_clipped_episode_rewards': sum(clipped_rewards),
                    }

                for tag, value in info.items():
                    log_value(tag, value, iteration)

            obs = env.reset()
            episode_rewards=[]
            clipped_rewards=[]
            last_lives = 0

        last_obs = preprocessing(obs)

        if(iteration>learning_starts and iteration%learning_freq == 0 ):

            idxs, obs_batch, act_batch, rew_batch, obs_next_batch, done_mask,ISweight = replay_buffer.sample(batchsize)

            obs_batch = torch.from_numpy(obs_batch).type(dtype=torch.float).to(device)
            act_batch = torch.from_numpy(act_batch).type(dtype=torch.long).to(device)
            rew_batch = torch.from_numpy(rew_batch).type(dtype=torch.float).to(device)
            obs_next_batch = torch.from_numpy(obs_next_batch).type(dtype=torch.float).to(device)
            done_mask = torch.from_numpy(done_mask).type(dtype=torch.float).to(device)
            ISweight = torch.from_numpy(ISweight).type(dtype=torch.float).to(device)
            # print(type(ISweight))

            # get the Q values for current observations (Q(s,a, theta_i))
            q_values = model(obs_batch)
            q_value = q_values.gather(1, act_batch.unsqueeze(1)).squeeze(1)
            # print(q_value)

            # next_q_state_values = Q_target(obs_next_batch)
            # # compute V*(next_states) using predicted next q-values
            # next_state_values =  next_q_state_values.max(-1)[0]  => next_q_value

            next_q_values = model(obs_next_batch)

            next_q_state_values = Q_target(obs_next_batch)
            
            next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)

            # Compute TD-error and update
            target_qvalues_for_actions = rew_batch + gamma * next_q_value * (1 - done_mask)

            errors = torch.abs(q_value - target_qvalues_for_actions).cpu().data.numpy()

            # update priority
            replay_buffer.update(idxs, errors)

            # MSE Loss function
            loss = (ISweight * F.mse_loss(q_value, target_qvalues_for_actions.detach())).mean()

            # backwards pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            for param in model.parameters():
                param.grad.data.clamp_(-1,1)
            # update
            optimizer.step()
            Qtarget_update_pointer += 1

            # update target Q network weights with current Q network weights
            if Qtarget_update_pointer % target_update_freq == 0:
                Q_target.load_state_dict(model.state_dict())

        # Log progress
        if iteration % save_model_flag == 0:

            if not os.path.exists("models"):
                os.makedirs("models")
            add_str = ''
            if (doubleDQN):
                add_str = 'double' 
            if (duelingDQN):
                add_str = 'dueling'

            model_save_path = "./models/%s_%d.model" %(add_str, iteration)
            torch.save(model.state_dict(), model_save_path)

        if iteration % log_flag == 0 :
            print("________________________________")
            print("Timestep %d" % (iteration,))
            print("Learning started? %d" % (iteration > learning_starts))
            print("Exploration %f" % threshold)
            sys.stdout.flush()

            if threshold>0 and log:
                info = {
                    'exploration': threshold,
                }

                for tag, value in info.items():
                    log_value(tag, value, iteration)

        iteration += 1

        if iteration % 1000==0 :
                print(iteration)


train(duelingDQN=True,render= True)
# test statement
# test(path = "./models/dueling_1200000.model",num_episodes = 1000,env=env,render = True,actionDim=actionDim,device=device,duelingDQN=True)
env.close()