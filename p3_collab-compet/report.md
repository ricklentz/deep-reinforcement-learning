## Deep Deterministic Policy Gradient - Actor-Critic, Continuous Control Task in Unity Reacher Env

[![20 Agent Reacher Environment](https://img.youtube.com/vi/ijF98-GBGqo/0.jpg)](https://www.youtube.com/watch?v=ijF98-GBGqo)

### Model Architecture
In designing the neural architecture, my strategy was based on several observations. The dominating observation is the general utility of the Actor-Critic model for solving a broad range of tasks.  Second, in non-synthetic environments, agents must assume a noise function while constructing an internal representation of the environment.  Whether direct observation or indirect construct formation through participation in an information sharing mechanism, this noise function can impact agent performance in unintuitive ways.  These observations resulted in a decision to decouple the agents in a manner to allow scaling (thousands) and to facilitate future participation in communications strategies (e.g. coaching, cumulative reward sharing, hierarchial directives, common evnironment broadcasts).



The Udacity provided actor and critic code in PyTorch as well as the noise function.  This code base was adapted for the 20 agent (version 2) environment. 

I used the actor-critic structure with each of the two hidden layers containing 400 and 300 nodes respectively.  ReLU activation is used on the hidden layers and tanh is used for the output layers. This architecture improves on the course's baseline performance provided as a starting point for this project.


### Hyperparameters
A learning rate of 1e-4 was used for each fully connected layer.  Tau value of .001 and Gamma of .99 were also used. Batch size was increased to 128 and the replay buffer size was left at 1e5.  Ornstein-Uhlenbeck noise parameters of 0.15 for Theta and 0.2 for Sigma were also left unchanged. These parameters were carried forward from the Actor-Critic Method lesson (https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal).


## Results and Future Work

This model is able to achieve the the reward performance goal of 30 in about 40 episodes.

<img src="score_episode_num.png" width="510" height="340" />

Architecture could be researched further.  Specifically, I'd like to use AdaNet (https://arxiv.org/abs/1607.01097) to see what different neural architectures provide performance improvements.  Interestingly, this would be a DRL system building a DRL Contineous Control System.  

Also, increasing the utility of the environment by perhaps increasing to six degrees of freedom + gripper. 

I could further investigate tuning the Ornstein-Uhlenbeck noise level.  These impact the degree of exploration the agent does.  Generally, my treatment of parameters could be improved by adopting recommendations of https://arxiv.org/pdf/1803.09820.pdf.
