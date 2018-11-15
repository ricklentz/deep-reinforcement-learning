from collections import deque

def ddpg(num_episodes=25000, max_t=1000, min_solve_threshold=.50, scores_window_length=100):
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
    # set the max scoring window length
    scores_window = deque(scores_window_length)
                  
    for i_episode in range(1, num_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]   
        states = env_info.vector_observations
        agent.reset()
        state = env_info.vector_observations
        score = np.zeros(num_agents)
        for t in range(max_t):
            # selection
            actions = list()
            for player in range(num_agents):
                actions.append( agent.act(states[player]) )
            # send the actions to the environment
            env_info = env.step(actions)[brain_name]
            # for each agent/player get the next state
            next_states = env_info.vector_observations
            # for each agent/player get the rewards
            rewards = env_info.rewards
            # check to see if we have met the goal
            dones = env_info.local_done
            
            states = next_states
            score += np.max(rewards)
            if np.any(dones):
                break 
        # save most recent score
        scores_window.append(score)
        all_scores.append(score)
        
        # save most recent score to list of all episode scores
        #scores.append(scores)              
        if (i_episode % 100 == 0):
            print('\rEpisode {}\tReward: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, np.mean(all_scores), np.mean(scores_window)))

        if np.mean(scores_window) >= min_solve_threshold:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
    return all_scores

