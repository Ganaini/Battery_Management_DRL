# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 13:29:02 2021

@author: mahmo
"""

from Env_DQN_MG_107 import Microgrid
from collections import namedtuple
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 256
BATCH_SIZE = 32
PERCENTILE = 70
ITERATIONS = 5000


class Net(nn.Module):
    def __init__(self, input_n, hidden_n, output_n):
        super(Net, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_n, hidden_n),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_n, hidden_n),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_n, hidden_n),
            nn.ReLU(),
            nn.Dropout(p=0.1),              
            nn.Linear(hidden_n, output_n)
            )
        
    def forward(self, x):
        return self.model(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

#generate batches with episodes
def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)

    while True:
        # time.sleep(0.001) # just to slow output screen
        obs_v = torch.FloatTensor([obs])
        # obs_v = obs_v.to(device)
        act_probs_v = sm(net(obs_v))
        # act_probs = act_probs_v.cpu().data.numpy()[0]
        act_probs = act_probs_v.data.numpy()[0]
        # np.random.seed(seed=0)
        action = np.random.choice(len(act_probs), p=act_probs)
        # make sure the action doesn't violate soc limits
        # if not env.action_accepted(action):
        #     # print('rejected:',action)
        #     continue
        # print(action)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        if is_done:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))

    train_obs_v = torch.FloatTensor(train_obs)
    # train_obs_v = train_obs_v.to(device)
    train_act_v = torch.LongTensor(train_act)
    # train_act_v = train_act_v.to(device)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = Microgrid()
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # net = Net(obs_size, HIDDEN_SIZE, n_actions)
    # net = net.to(device)
 
    # load state dict
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    PATH = r'E:\Mahmoud\PhD_Work\Presentations\March_25_Forecast\Models\A_day_trained_Normalized_103.pth'
    net.load_state_dict(torch.load(PATH))
    
    # or load all
    # PATH = r'E:\Mahmoud\PhD_Work\Presentations\Feb_8th\Model\april_all_165426.pth'
    # net = torch.load(PATH) 

    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.001)
    writer = SummaryWriter(comment="Microgrid")

    for iter_no, batch in enumerate(iterate_batches(
            env, net, BATCH_SIZE), start=0):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        # print(acts_v)    
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.3f, rw_bound=%.3f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        writer.add_scalar("actions", acts_v.numpy()[0], iter_no)
        
        if iter_no >= ITERATIONS or loss_v.item() < 0.0003:
            print('\n',obs_v) 
            print('\n',acts_v)
            print('\n',action_scores_v)
            print('\n',reward_b)
            print('\n',reward_m)
            print("FINISHED!")
            break
        
    # Save Model
    # save_path = r'E:\Mahmoud\PhD_Work\Presentations\Feb_8th\Model\July_dict_2800kW.pth'
    # torch.save(net.state_dict(), save_path)
        
    writer.close()
