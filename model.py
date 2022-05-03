import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
       def __init__(self,num_actions):
           super(DQN, self).__init__()

           self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
            )
           self.fc = nn.Sequential(
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
            )


       def forward(self, x):
            x = self.features(x)
            # print(x.size())
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x.float()

class DuelingNetwork(nn.Module):
    def __init__(self, num_actions):
        super(DuelingNetwork, self).__init__()
        self.num_actions = num_actions
        
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1_advantage = nn.Linear(in_features=7*7*64, out_features=512)
        self.fc1_value = nn.Linear(in_features=7*7*64, out_features=512)

        self.fc2_advantage = nn.Linear(in_features=512, out_features=num_actions)
        self.fc2_value = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        # print(x.size())
        x = self.relu(self.conv2(x))
        # print(x.size())
        x = self.relu(self.conv3(x))
        # print(x.size())
        x = x.view(x.size(0), -1)

        advantage = self.relu(self.fc1_advantage(x))
        value = self.relu(self.fc1_value(x))

        advantage = self.fc2_advantage(advantage)
        value = self.fc2_value(value)

        x = value + advantage - advantage.mean()

        return x