
from unityagents import UnityEnvironment
from agent import agent
import numpy as np

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
        
       
        # set the agents
        self.agent_list = list()
        
    def ready_agents(self, display_info=True, train_mode=True):
        # get the default brain
        self.brain_name = self.env.brain_names[0]
        brain = self.env.brains[self.brain_name]
        
        if display_info==True:
            print("Brain name " + self.brain_name)
        
        # refresh the environment
        self.env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        
        # number of agents
        self.num_agents = len(self.env_info.agents)
        if display_info==True:
            print('Number of agents:', self.num_agents)
        
        # size of each action
        self.action_size = brain.vector_action_space_size
        if display_info==True:
            print('Size of each action:', self.action_size)
        
        # examine the state space 
        states = self.env_info.vector_observations
        self.state_size = states.shape[1]
        if display_info==True:
            print('Each agent observes a stacked (x3) state of total length: {}'.format( self.state_size))
        
        # display the vector space
        if display_info==True:
            for idx in range(self.num_agents):
                print('The state for the agent ' + str(idx) + ' looks like: ', states[idx])
        
        # add each agents specific parameters
        for idx in range(self.num_agents):
            self.agent_list.append( agent(state_size=self.state_size, action_size=self.action_size ))
                                   
    def run_ddpg(self, n_episodes=6500, max_timesteps=10000, min_solve_threshold=0.50, scores_window_length=100):
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

        for i_episode in range(1, n_episodes+1):
            self.env_info = self.env.reset(train_mode=True)[self.brain_name]  
            states = self.env_info.vector_observations 
            #print(states.shape)
            #print(str(self.state_size * self.num_agents))
            #states = np.reshape(states, (1,48))
            #print(states.shape)
            for idx, agent in enumerate( self.agent_list ):
                agent.reset()
            scores = np.zeros(self.num_agents)
            while True: #for t in range(max_timesteps):
                # evaluate actions
                #print(str(t) + ' ' + str(states.shape))
                #print(str(t) + ' ' + str(states[0].shape))
                actions = list()
                for idx, agent in enumerate( self.agent_list ):
                    action = agent.act(np.reshape(states[idx],(1,self.state_size)))
                    actions.append(action)
                    #print(str(t) + ' ' + str(states[1].shape))
                    #act_1 = self.agent_list[1].act(np.reshape(states[1],(1,24)))          
                    
                    #print(actions.shape)
                    #actions = np.reshape(actions, (1, 4))
                    #print('Actions')
                    #print(actions)
                    # this environment is common to multiple agents, play it forward one step
                #actions_shaped = np.reshape(np.concatenate((actions[0],actions[1]),axis=0), (1, 4))
                self.env_info = self.env.step(np.concatenate((actions[0],actions[1]),axis=0))[self.brain_name]
                    
                #print('Env Info')
                #print(env_info)
                # now observe the next state for each action                   
                #next_states = self.env_info.vector_observations
                next_states = self.env_info.vector_observations         # get next states
                #next_states = np.reshape(next_states, (1, 48))     # combine each agent's state into one st
                #print('next states')
                #print(next_states)
                # the actions result in a set of awards
                rewards = self.env_info.rewards
                #print('rewards')
                #print(rewards)
                # the solution may also be met by actions
                dones = self.env_info.local_done 
                #print('dones')
                #print(dones)
                # given information afout the state space, reward, actions, etc learn about the stochastic process
                #actions = np.concatenate(actions, axis=0) 
                for idx, agent in enumerate( self.agent_list ):
                    #print(states[idx].shape)
                    #print(actions[idx].shape)
                    #print(actions[idx])
                    #print(rewards[idx])
                    #print(next_states[idx].shape)
                    #print(dones[idx])
                    agent.step(states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx])
            
                # tally results
                scores += rewards
            
                # env changes as time passes, future expectations are now 
                states = next_states

                # end is needed
                if np.any(dones):
                    break 

                # save most recent score
                scores_window.append(np.max(scores))
                all_scores.append(np.max(scores))

            if (i_episode % 10 == 0):
                print('\rEpisode {}\tMax Reward: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, np.max(scores), np.mean(scores_window)))
                #for idx, agent in enumerate( self.agent_list ):
                    #print(agent.timestep)

            if np.mean(scores_window) >= min_solve_threshold:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                for idx, agent in enumerate( self.agent_list ):
                    torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_agent'+str(idx)+'.pth')
                    torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_agent'+str(idx)+'.pth')
                break
        return all_scores 

