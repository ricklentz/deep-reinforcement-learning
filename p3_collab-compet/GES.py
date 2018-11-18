
from unityagents import UnityEnvironment
import numpy as np
from agent import MultiAgentDeepDeterministicPolicyGradient
import torch


import time
from collections import deque
import numpy as np



class Stats():
    def __init__(self):
        self.score = None
        self.avg_score = None
        self.std_dev = None
        self.scores = []                         # list containing scores from each episode
        self.avg_scores = []                     # list containing average scores after each episode
        self.scores_window = deque(maxlen=100)   # last 100 scores
        self.best_avg_score = -np.Inf            # best score for a single episode
        self.time_start = time.time()            # track cumulative wall time
        self.total_steps = 0                     # track cumulative steps taken

    def update(self, steps, rewards, i_episode):
        """Update stats after each episode."""
        self.total_steps += steps
        self.score = sum(rewards)
        self.scores_window.append(self.score)
        self.scores.append(self.score)
        self.avg_score = np.mean(self.scores_window)
        self.avg_scores.append(self.avg_score)
        self.std_dev = np.std(self.scores_window)
        # update best average score
        if self.avg_score > self.best_avg_score and i_episode > 100:
            self.best_avg_score = self.avg_score

    def is_solved(self, i_episode, solve_score):
        """Define solve criteria."""
        return self.avg_score >= solve_score and i_episode >= 100

    def print_episode(self, i_episode, steps, stats_format, buffer_len, noise_weight,
                      critic_loss_01, critic_loss_02,
                      actor_loss_01, actor_loss_02,
                      noise_val_01, noise_val_02,
                      rewards_01, rewards_02):
        common_stats = 'Episode: {:5}   Avg: {:8.3f}   BestAvg: {:8.3f}   σ: {:8.3f}  |  Steps: {:8}   Reward: {:8.3f}  |  '.format(i_episode, self.avg_score, self.best_avg_score, self.std_dev, steps, self.score)
        print('\r' + common_stats + stats_format.format(buffer_len, noise_weight), end="")
       

    def print_epoch(self, i_episode, stats_format, *args):
        n_secs = int(time.time() - self.time_start)
        common_stats = 'Episode: {:5}   Avg: {:8.3f}   BestAvg: {:8.3f}   σ: {:8.3f}  |  Steps: {:8}   Secs: {:6}      |  '.format(i_episode, self.avg_score, self.best_avg_score, self.std_dev, self.total_steps, n_secs)
        print('\r' + common_stats + stats_format.format(*args))

    def print_solve(self, i_episode, stats_format, *args):
        self.print_epoch(i_episode, stats_format, *args)
        print('\nSolved in {:d} episodes!'.format(i_episode-100))

class general_environment_solver():
    """ General Solver for Unity Environments """
    def __init__(self, unity_env='Tennis.app'):
        
        """Initialize a general environment solver.  Run refresh_env to reset the solver.
        You can run the solver using the defaults.
        
        Params
        ======
            unity_env (string): path to unity environment
        """
        self.env = UnityEnvironment(file_name=unity_env)
        self.train_mode = True
       
        # set the agents
        self.agent_list = list()
        
    def ready_agents(self, display_info=True, train_mode=True):
        # get the default brain
        self.brain_name = self.env.brain_names[0]
        brain = self.env.brains[self.brain_name]
        
        if display_info==True:
            print("Brain name " + self.brain_name)
        
        # refresh the environment
        self.env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        
        # number of agents
        self.num_agents = len(self.env_info.agents)
        if display_info==True:
            print('Number of agents:', self.num_agents)
        
        # size of each action
        self.action_size = brain.vector_action_space_size
        if display_info==True:
            print('Size of each action:', self.action_size)
        
        # examine the state space 
        self.state = self.env_info.vector_observations
        
        self.agent = MultiAgentDeepDeterministicPolicyGradient()
                                   
    def run_maddpg(self, n_episodes=6500, max_timesteps=1000, min_solve_threshold=0.50, scores_window_length=100):
        # train the agent
        from collections import deque
        """ MultiAgent Deep Deterministic Policy Gradients
    
        Params
        ======
            num_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            min_solve_threshold (float): a score metric that result in training to stop
            scores_window_length (int): maximum length of scoring metric window, e.g. last 100 scores
         """
    
        # list containing scores from each episode
        all_scores = []

        scores_window = deque(maxlen=scores_window_length)
        stats = Stats()
        stats_format = 'Buffer: {:6}   NoiseW: {:.4}'

        for i_episode in range(1, n_episodes+1):
            rewards = []
            self.env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
            self.state = self.env_info.vector_observations

            # loop over steps
            for t in range(max_timesteps):
                # select an action
                if self.agent.evaluation_only:  # disable noise on evaluation
                    action = self.agent.act(self.state, add_noise=False)
                else:
                    action = self.agent.act(self.state)

                # take action in environment
                self.env_info = self.env.step(action)[self.brain_name]
                next_state = self.env_info.vector_observations
                reward = self.env_info.rewards
                done = self.env_info.local_done

                # update agent with returned information
                self.agent.step(self.state, action, reward, next_state, done)
                self.state = next_state
                rewards.append(reward)
                if any(done):
                    break

            # every episode
            buffer_len = len(self.agent.memory)
            per_agent_rewards = []  # calculate per agent rewards
            for i in range(self.agent.num_agents):
                per_agent_reward = 0
                for step in rewards:
                    per_agent_reward += step[i]
                per_agent_rewards.append(per_agent_reward)
            stats.update(t, [np.max(per_agent_rewards)], i_episode)  # use max over all agents as episode reward
            stats.print_episode(i_episode, t, stats_format, buffer_len, self.agent.noise_weight,
                                self.agent.agents[0].critic_loss, self.agent.agents[1].critic_loss,
                                self.agent.agents[0].actor_loss, self.agent.agents[1].actor_loss,
                                self.agent.agents[0].noise_val, self.agent.agents[1].noise_val,
                                per_agent_rewards[0], per_agent_rewards[1])

            # every epoch (100 episodes)
            if i_episode % 100 == 0:
               #stats.print_epoch(i_episode, stats_format, buffer_len, agent.noise_weight)
               save_name = 'saves/episode.{}.'.format(i_episode)
               for i, save_agent in enumerate(self.agent.agents):
                   torch.save(save_agent.actor_local.state_dict(), save_name + str(i) + '.actor.pth')
                   torch.save(save_agent.critic_local.state_dict(), save_name + str(i) + '.critic.pth')

            # if solved
            if stats.is_solved(i_episode, min_solve_threshold):
                stats.print_solve(i_episode, stats_format, buffer_len, agent.noise_weight)
                save_name = 'saves/solved.'
                for i, save_agent in enumerate(agent.agents):
                    torch.save(save_agent.actor_local.state_dict(), save_name + str(i) + '.actor.pth')
                    torch.save(save_agent.critic_local.state_dict(), save_name + str(i) + '.critic.pth')
                break


