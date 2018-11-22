## MultiAgent Deep Deterministic Policy Gradient - Actor-Critic, Continuous Control Task in Unity Tennis Environment



### Model Architecture


The Udacity provided actor and critic code in PyTorch as well as the noise function.  This code base was adapted from the two agent Tennis environment. 

This neural architecture has two DDPG agents (https://arxiv.org/abs/1509.02971) with identical but distinct Actor-Critic models.  The difference between this structure and the prior project is the use of a shared replay buffer for the Critic.  The effect of this implementation is that the Critic has access to both Agent's experience as published in 
Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (https://arxiv.org/abs/1706.02275)  The Actor has two fully connected hidden layers, with 350 and 300 nodes, and tanh is used for the output layer. The Critic has two fully connected hidden layers, each with 350 nodes, using relu activation for both hidden layers. This architecture improves on the course's baseline performance provided as a starting point for this project.

Todo:
1) use batch normalization torch.nn.modules.batchnorm
2) use parameter space noise rather than noise on action (https://vimeo.com/252185862)

3) use prioritised experience buffer (https://github.com/Damcy/prioritized-experience-replay) distinct to each Actor/Critic

4) Add dropout in the critic network

5) Turn off OU noise and use random noise




### Hyperparameters
Hyperparameters, I used a larger batch size (256 vs. 64) and a smaller buffer size, 100k vs. 1M; a noise decay of 0.999999 while starting the noise multiplier at 1.1999; while doubling the learning rate of the Actor to  2e-4.  I found that increasing tau by order an order of magnitude, to 1e-2, helped reduce training time and I held Gamma at 0.99.  The Critic's learning rate is 5e-4, and I zeroed the Critic's weight decay.  The given Ornstein-Uhlenbeck noise function parameters of mu=0.0, theta=0.15, and sigma=0.2 are held constant.

## Results and Future Work

This model is able to achieve the the reward performance goal of 0.5 in just under 24,000 episodes.

<img src="score_episode_num.png" width="1494" height="751" />

Architecture could be researched further.  Specifically, I'd like to use the directions mentioned in the MAACMCCE paper, it would be nice to test an ensemble of policy networks.  Also, working to restrict the environment information available to Critic would be helpful before attempting the Soccer Environment and perhaps for other more real world applications.  

Also, it would be good to investigate tuning the Ornstein-Uhlenbeck noise level.  These impact the degree of exploration the agents may result in a more diverse range of input experiences.  
