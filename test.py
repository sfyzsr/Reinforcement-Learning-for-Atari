import gym
import numpy as np
import torch
import cv2

import random
import os

from model import DQN, DuelingNetwork


def save_frames_as_video(frames, fileName="1.avi"):
    # Save frame list to video
    if not os.path.exists("videos"):
                os.makedirs("videos")

    height, width, layers = frames[0].shape

    out = cv2.VideoWriter("./videos/"+fileName,0, 90, (width,height))

    for i in range(len(frames)):
        out.write(frames[i])
    cv2.destroyAllWindows()
    out.release()

def preprocessing(image):
# Preprocssing the image and reduce the dimensionality from 210*160*3 RGB to 84*84*1 Gray scale
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    # print(np.shape(gray))   #210*160
    DSgray = cv2.resize(gray,(84,110))
    # CDSgray = DSgray[:84,:]    #84*84     Use this clipping method, if env is MsPacman-v0
    CDSgray = DSgray[26:,:]     # Use this clipping, if env is Breakout-v0
    img = np.reshape(CDSgray, [84, 84, 1])
    return img.astype(np.uint8)

def test(path, num_episodes,env,actionDim,device,doubleDQN=False, duelingDQN=False,render = False):
# Test function: test model performance
    # path: saved model path
    # num_episodes: number of episodes to test
    # env: gym environment
    # actionDim: action Dimenison
    # device: torch device
    # doubleDQN: used to select the network if use then True
    # duelingDQn: used to select the network if use then True
    # render: render the game when test

    # Set the model and optimizer
    assert  doubleDQN or duelingDQN , "Must set one Q function"
    if doubleDQN:
        Q_Func = DQN
    if duelingDQN:
        Q_Func = DuelingNetwork

    model = Q_Func(actionDim).float().to(device)

    model.load_state_dict(torch.load(path))
    model.eval()

    avg_reward = 0
    highest_episode_reward = 0
    highest_clipped_reward = 0
    
    for episode in range(num_episodes):
        
        s0 = env.reset()
        
        frames = []
        fileName = ""
        frames.append(s0)

        s0 = preprocessing(s0)
        s0 = torch.from_numpy(s0).type(dtype=torch.float).cuda()
        # print(s0.shape)
        s0 = s0.permute(2, 0, 1) 
        state = []

        episode_reward = 0
        clipped_reward = 0

        # Random act three step to get the frames (4,84,84)
        for _ in range(3):
            action = env.action_space.sample()
            s, r, done, info = env.step(action)

            episode_reward += r
            clipped_reward += np.clip(r,-1,1)
            frames.append(s)

            s = preprocessing(s)
            s = torch.from_numpy(s).type(dtype=torch.float).cuda()
            s = s.permute(2, 0, 1) 
            state.append(s)

        state = torch.cat((s0, state[0], state[1], state[2])).unsqueeze(0)
        episode_steps = 0
        done = False
        
        while not done:

            if render:
                env.render() 

            if np.random.random() < 0.1:
                action = random.randrange(actionDim)
                
            else:
                action = model(state).cuda()
                # print(action)
                action = action.max(1)[1].item()

            s1, reward, done, info = env.step(action)

            frames.append(s1)

            s1 = preprocessing(s1)
            s1 = torch.from_numpy(s1).type(dtype=torch.float).cuda()
            s1 = s1.permute(2, 0, 1) 

            s1 = torch.cat((state.squeeze(0)[1:, :, :], s1)).unsqueeze(0)

            state = s1
            episode_reward += reward
            clipped_reward += np.clip(reward,-1,1)
            episode_steps += 1

        avg_reward += episode_reward
        # only save the good result
        if(episode_reward>highest_episode_reward):
            highest_episode_reward = episode_reward
            fileName = "episodeReward"+str(episode_reward)+"_"+str(episode)+".avi"
            save_frames_as_video(frames=frames,fileName=fileName)

        if(clipped_reward>highest_clipped_reward):
            highest_clipped_reward = clipped_reward
            fileName = "clippedReward"+str(clipped_reward)+"_"+str(episode)+".avi"
            save_frames_as_video(frames=frames,fileName=fileName)

        if(episode%10==0):
            print(episode)
    avg_reward /= num_episodes
    print("total reward: %5f" % (avg_reward))

# env = gym.make('Breakout-v0')
# last_obs = env.reset()


# plt.imshow(preprocessing(last_obs))
# plt.show()