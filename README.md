# DRL-DeepQNetwork-training-failures-summarise
This repository analyzes several possible reasons for leading false positive training in DQN with CartPole (gymnasium). \
**Briefly speaking**, the best way to figure out your script falls into the false positive case rather than false configuration or misunderstanding in DQN mechanisms is to **train a pre-trained** model in the same case.

## Concepts
* False positive training: the model converges to a low reward value due to non-fine-tuned hyperparameter settings, like this:
![False Positive training](images/training_from_random.png) \
* Target Model (/Policy/network): the model that generates temporal difference target (TD target, or target Q value: r + Q(s_(t+1))), almost with froze parameter (for stabilization and convergence speed).
* Action Model (/Policy/network) or Policy_Net: the model that is updating frequently with stepping.
* Q: estimation value to a given state (or station-action pair).

## DQN is sensitive to hyperparameter settings and can be skewed easily:
* Batch Size and learning Rate's influences: [DeepReinforcementLearning_DQN_Test.ipynb](DeepReinforcementLearning_DQN_Test.ipynb)
* If the model is not instructed in the proper direction, it's highly possible to be led into false positive: \
  For the same hyperparameters configuration, as shown in [Nov_05_2024_DQN_test.ipynb](Nov_05_2024_DQN_test.ipynb): \
  Model initialized randomly: \
  ![False Positive training](images/training_from_random.png) \
  Model initialized with pre-trained parameters ([DQN_official.pt](DQN_official.pt) from [pytorch](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)) with skewed reward (terminated reward value 0 -> -5): \
  ![images/training_from_pre-trained_skewed.png](images/training_from_pre-trained_skewed.png)

Reference: 
1. [Pytorch Implementation](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#training)
2. [Tensorflow Implementation GreeksForGreeks](https://www.geeksforgeeks.org/a-beginners-guide-to-deep-reinforcement-learning/)
3. [A simple introduction of deep reinforcement learning](https://aws.amazon.com/what-is/reinforcement-learning/#:~:text=Reinforcement%20learning%20(RL)%20is%20a,use%20to%20achieve%20their%20goals.)
4. A survey article may help with further studying: [Deep Reinforcement Learning: A Survey](https://ieeexplore.ieee.org/document/9904958)
5. An implementation of DRL from one of my known professors: [Optimizing Data Center Energy Efficiency via Event-Driven Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/9729602)
6. Effable explanation of special operator asterisk *: https://stackoverflow.com/questions/19339/transpose-unzip-function-inverse-of-zip/19343#19343
7. Map() function: https://www.geeksforgeeks.org/python-map-function/
8. Lambda() function: https://www.geeksforgeeks.org/how-to-use-if-else-elif-in-python-lambda-functions/


Detailed explanations are inside the [DeepReinforcementLearning_DQN_Test.ipynb](https://github.com/TyBruceChen/Deep-Reinforcement-Learning-with-Deep-Q-Network--a-simple-implementation/blob/main/DeepReinforcementLearning_DQN_Test.ipynb) file.
