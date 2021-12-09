import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha,  fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/'):
        super(CriticNetwork, self).__init__()
        
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_ppo')
        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
            )
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cudo:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        value = self.critic(state)
        return value
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))