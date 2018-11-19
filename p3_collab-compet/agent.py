import torch
import torch.optim as optim
import random
import torch.nn.functional as F

import numpy as np
from collections import namedtuple, deque
import copy


from model import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiAgentDeepDeterministicPolicyGradient():
    """Interacts with and learns from the environment using multiple agents."""
    def __init__(self, action_size=2, seed=1999, load_file=None,num_agents=2,buffer_size=int(1e5),batch_size=256,gamma=0.99,update_every=4,noise_weight=1.1999,noise_decay=.999999,evaluation_only=False):
        """
        Params
        ======
            action_size (int): dimension of each action
            seed (int): Random seed
            load_file (str): path of checkpoint file to load
            num_agents (int): number of distinct agents
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            noise_start (float): initial noise weighting factor
            noise_decay (float): noise decay rate
            update_every (int): how often to update the network
            evaluation_only (bool): set to True to disable updating gradients and adding noise
        """
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        self.num_agents = num_agents
        self.noise_weight = noise_weight
        self.noise_decay = noise_decay
        self.timestep = 0
        self.evaluation_only = evaluation_only
        
        # create two agents, each with their own actor and critic (but shared memory/experience buffer)
        self.agents = [DeepDeterministicPolicyGradientAgent( 0, batch_size, gamma, seed), DeepDeterministicPolicyGradientAgent( 1, batch_size, gamma, seed)]
        # create shared replay buffer
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        if load_file:
            for i, save_agent in enumerate(self.agents):
                actor_file = torch.load(load_file + '.' + str(i) + '.actor.pth', map_location='cpu')
                critic_file = torch.load(load_file + '.' + str(i) + '.critic.pth', map_location='cpu')
                save_agent.critic_local.load_state_dict(critic_file)
                save_agent.critic_target.load_state_dict(critic_file)
                save_agent.actor_local.load_state_dict(actor_file)
                save_agent.actor_target.load_state_dict(actor_file)
            print('Loaded: {}.critic.pth'.format(load_file))    
            print('Loaded: {}.actor.pth'.format(load_file))
            

    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones):
        # reshape 2x24 into 1x48 dim vector
        all_states = all_states.reshape(1, -1)  
        # reshape 2x24 into 1x48 dim vector
        all_next_states = all_next_states.reshape(1, -1)  
        self.memory.add(all_states, all_actions, all_rewards, all_next_states, all_dones)
        # Learn every update_every time steps.
        self.timestep = (self.timestep + 1) % self.update_every
        if self.timestep == 0 and self.evaluation_only == False:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                # each agent does it's own sampling from the replay buffer
                experiences = [self.memory.sample() for _ in range(self.num_agents)]
                self.learn(experiences, self.gamma)

    def act(self, all_states, add_noise=True):
        # pass each agent's state from the environment and calculate it's action
        all_actions = []
        for agent, state in zip(self.agents, all_states):
            action = agent.act(state, noise_weight=self.noise_weight, add_noise=True)
            self.noise_weight *= self.noise_decay
            all_actions.append(action)
        return np.array(all_actions).reshape(1, -1) # reshape 2x2 into 1x4 dim vector

    def learn(self, experiences, gamma):
        # each agent uses it's own actor to calculate next_actions
        all_next_actions = []
        for i, agent in enumerate(self.agents):
            _, _, _, next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            next_state = next_states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            next_action = agent.actor_target(next_state)
            all_next_actions.append(next_action)
        # each agent uses it's own actor to calculate actions
        all_actions = []
        for i, agent in enumerate(self.agents):
            states, _, _, _, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            action = agent.actor_local(state)
            all_actions.append(action)
        # each agent learns from it's experience sample
        for i, agent in enumerate(self.agents):
            agent.learn(i, experiences[i], gamma, all_next_actions, all_actions)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        random.seed(seed)

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)






class DeepDeterministicPolicyGradientAgent():
    """Interacts with and learns from the environment."""
    def __init__(self, id, batch_size, gamma, seed, state_size=24, action_size=2, critic_observable_agent_envs=2):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        random.seed(seed)
        np.random.seed(seed)

        self.id = id
        self.action_size = action_size
        self.state_size = state_size
        self.seed = random.seed(seed)
        self.critic_observable_agent_envs = critic_observable_agent_envs
        self.batch_size = batch_size        # minibatch size
        self.gamma = gamma            # discount factor
        self.tau = 1e-2             # for soft update of target parameters
        self.lr_actor = 1e-4         # learning rate of the actor
        self.lr_critic = 1e-3        # learning rate of the critic
        self.critic_weight_decay = 0.0  #0.00001   # L2 weight decay

        # track stats for tensorboard logging
        self.critic_loss = 0
        self.actor_loss = 0
        self.noise_val = 0
       
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic( state_size, action_size, critic_observable_agent_envs, seed).to(device)
        self.critic_target = Critic(state_size, action_size, critic_observable_agent_envs, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.critic_weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, seed)


    def act(self, state, noise_weight=1.0, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        # calculate action values
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            self.noise_val = self.noise.sample() * noise_weight
            action += self.noise_val
        return np.clip(action, -1, 1)


    def reset(self):
        self.noise.reset()

    def learn(self, agent_id, experiences, gamma, all_next_actions, all_actions):
        """Update policy and value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            all_next_actions (list): each agent's next_action (as calculated by it's actor)
            all_actions (list): each agent's action (as calculated by it's actor)
        """

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # get predicted next-state actions and Q values from target models
        self.critic_optimizer.zero_grad()
        agent_id = torch.tensor([agent_id]).to(device)
        actions_next = torch.cat(all_next_actions, dim=1).to(device)
        with torch.no_grad():
            q_targets_next = self.critic_target(next_states, actions_next)
        # compute Q targets for current states (y_i)
        q_expected = self.critic_local(states, actions)
        # q_targets = reward of this timestep + discount * Q(st+1,at+1) from target network
        q_targets = rewards.index_select(1, agent_id) + (gamma * q_targets_next * (1 - dones.index_select(1, agent_id)))
        # compute critic loss
        critic_loss = F.mse_loss(q_expected, q_targets.detach())
        self.critic_loss = critic_loss.item()  # for tensorboard logging
        # minimize loss
        critic_loss = critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # compute actor loss
        self.actor_optimizer.zero_grad()
        # detach actions from other agents
        actions_pred = [actions if i == self.id else actions.detach() for i, actions in enumerate(all_actions)]
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_loss = actor_loss.item()  # calculate policy gradient
        # minimize loss
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.actor_local, self.actor_target, self.tau)
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        
            

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        random.seed(seed)
        np.random.seed(seed)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

    