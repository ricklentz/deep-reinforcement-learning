import torch
import torch.optim as optim
import random
import torch.nn.functional as F
from actor import actor
from critic import critic
from ornstein_uhlenbeck_noise import ornstein_uhlenbeck_noise
from replay_buffer import replay_buffer
import numpy as np

class agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, eps_start=6,eps_end=0,eps_decay=256,random_seed=1999, critic_weight_decay = 0, learning_rate_actor = 5e-3, learning_rate_critic = 5e-3, replay_buffer_size= int(1e6), mini_batch_size=128, gamma=.99, tau=6e-2 ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            critic_weight_decay (float): Critic weight decay parameter used for Adam Optimizer 
        """
        # Torch hardare device specifics
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.mini_batch_size = mini_batch_size

        self.eps = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        # Actor Network (w/ Target Network)
        self.actor_local = actor(state_size, action_size, random_seed).to(self.device)
        self.actor_target = actor(state_size, action_size, random_seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = critic(state_size, action_size, random_seed).to(self.device)
        self.critic_target = critic(state_size, action_size, random_seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=learning_rate_critic, weight_decay=critic_weight_decay)

        # Noise process
        self.noise = ornstein_uhlenbeck_noise((1, action_size), random_seed)

        # Replay memory
        self.memory = replay_buffer(action_size, replay_buffer_size, mini_batch_size, random_seed, self.device)

        self.timestep = 0
        self.update_every = 1
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / rewards
        self.memory.add(state, action, reward, next_state, done)
        
        self.timestep = (self.timestep + 1) % self.update_every
        if self.timestep == 0:
            # each player has to learn from its own local observations, if enough samples are available in memory
            if len(self.memory) > self.mini_batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, states, add_noise=True):
        #print(states.shape)
        """Returns actions for given state as per current policy."""
        # convert states format to tensor
        states = torch.from_numpy(states).float().to(self.device)
        actions = np.zeros((1, self.action_size))
        self.actor_local.eval()
        
        with torch.no_grad():
            for state_num_num, state in enumerate(states):
                action = self.actor_local(state).cpu().data.numpy()
                actions[state_num_num, :] = action
        self.actor_local.train()
        if add_noise:
            actions = self.eps * self.noise.sample()
        return np.clip(actions, -1, 1)
    
    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        #if self.agent_num == 0:
        #     actions_next = torch.cat((actions_next, actions[:,2:]), dim=1)
        # else:
        #     actions_next = torch.cat((actions[:,:2], actions_next), dim=1)

        #print(next_states.shape)
        #print(actions_next.shape)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        #print(rewards.shape)
        #print(Q_targets_next.shape)
        #print(dones.shape)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)

        # if self.agent_num == 0:
        #     actions_pred = torch.cat((actions_pred, actions[:,2:]), dim=1)
        # else:
        #     actions_pred = torch.cat((actions[:,:2], actions_pred), dim=1)

        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)     

        self.eps = self.eps - (1/self.eps_decay)
        if self.eps < self.eps_end:
            self.eps=self.eps_end
            

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(),local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def reset(self):
        self.noise.reset()
    