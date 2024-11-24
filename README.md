# DRL-DeepQNetwork-training-failures-summarise
This repository analyzes several possible reasons for leading false positive training in DQN with CartPole (gymnasium). \
**Briefly speaking**, the best way to figure out your script falls into the false positive case rather than false configuration or misunderstanding in DQN mechanisms is to **train a pre-trained** model in the same case.
The influential factors are: 
* [penalty reward](https://github.com/gasaiginko/Deep-Reinforcement-Learning-with-Deep-Q-Network--a-simple-implementation#different-penalty-reward-configurations-contribution-to-training-results): not clearly distinguished from the normal reward
* [learning rate](https://github.com/gasaiginko/Deep-Reinforcement-Learning-with-Deep-Q-Network--a-simple-implementation#learning-rate-is-significant-to-the-training-time-for-achieving-an-acceptable-model): not enough training episodes
* [batch size](https://github.com/gasaiginko/Deep-Reinforcement-Learning-with-Deep-Q-Network--a-simple-implementation#large-batch-size-can-stabilize-the-result): too small -> unstable
* [clamp value](https://github.com/gasaiginko/Deep-Reinforcement-Learning-with-Deep-Q-Network--a-simple-implementation#clamp-gradient-can-lead-to-a-stable-training-process-and-better-results)

## Concepts
* False positive training: the model converges to a low reward value due to non-fine-tuned hyperparameter settings, like this: \
![False Positive training](images/training_from_random.png) 
* Target Model (/Policy/network): the model that generates temporal difference target (TD target, or target Q value: r + Q(s_(t+1))), almost with froze parameter (for stabilization and convergence speed).
* Action Model (/Policy/network) or Policy_Net: the model that is updating frequently with stepping.
* Q: estimation value to a given state (or station-action pair).

## DQN is sensitive to hyperparameter settings and can be skewed easily:
* Batch Size and learning Rate's influences: [DeepReinforcementLearning_DQN_Test.ipynb](DeepReinforcementLearning_DQN_Test.ipynb)
* If the model is not instructed in the proper direction, it's highly possible to be led into false positive: \
  For the same hyperparameters configuration, as shown in [Nov_05_2024_DQN_test.ipynb](Nov_05_2024_DQN_test.ipynb): \
  Model initialized randomly: \
  ![False Positive training](images/training_from_random.png) \
  Model initialized with pre-trained parameters ([DQN_official.pt](DQN_official.pt) from [pytorch](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)) with skewed reward (terminated reward (penalty) value 0 -> -5): \
  ![images/training_from_pre-trained_skewed.png](images/training_from_pre-trained_skewed.png) \
  Thus a well configuration is required for training from random initialization!

## Env specification: [gymnasium/cartpole_v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) 
Reward: 1 -> stay in 'balanced' threshold; 0 -> termination (out of 'balanced threshold') \

## Different penalty reward configurations' contribution to training results:
Batch size 4, Learning rate 1e-2, termination penalty: 0, clamp threshold: (-1,1) \
![training_from_random_bs4_lr1e-2_with_0penalty](https://github.com/user-attachments/assets/8b3d7edd-d463-4f69-90d8-e5ef426646fa) \
Batch size 4, learning rate 1e-2, termination penalty: -15, clamp threshold: (-1,1) \
![training_from_random_bs4_lr1e-2](https://github.com/user-attachments/assets/5e8c2d7d-a529-4622-b1ea-f2d444aa22b0)

## Learning rate is significant to the training time for achieving an acceptable model: 
A large learning rate can visualize results faster. \
Batch size 4, learning rate 1e-2, termination penalty: -15, clamp threshold: (-1,1) -> fit at 250 ep \
![training_from_random_bs4_lr1e-2](https://github.com/user-attachments/assets/d918bf9b-22ec-4bfa-ba9a-7a15f4ba5f53) \
Same configuration with changing learning rate to 1e-3 -> fit at 300 ep \
![training_from_random_bs4_lr1e-3](https://github.com/user-attachments/assets/02d6fe04-7d09-4886-b969-f27ca6a3abbe) \
Same configuration with changing learning rate to 1e-4 -> fit at 850 ep \
![training_from_random_bs4_lr1e-4](https://github.com/user-attachments/assets/9385ec95-b20f-4769-83ce-65e753b19b74)

## Large Batch size can stabilize the result: 

With different batch sizes {4,32,128,256} under the same hyperparameter configuration: \
![training_from_random_bs4_lr1e-3](https://github.com/user-attachments/assets/56bc1c9a-9258-49d9-9ed9-5783f6bb00a9)
![training_from_random_bs32_lr1e-3](https://github.com/user-attachments/assets/8c1b34e8-e077-449b-bfba-e4920dcc8bde)
![training_from_random_bs128_lr1e-3](https://github.com/user-attachments/assets/83837396-4987-4ff5-b25f-85e3859115e5)
![training_from_random_bs256_lr1e-3](https://github.com/user-attachments/assets/d8ee691a-c663-4f2a-9b58-5baad1d9b2c6)


## Clamp gradient can lead to a stable training process and better results: 
(Gradient) clamp: enforce the values of gradients to exist in a specific range. Exp.: lambda g: max(min(g, max_value), min_value) \
With different clamp abs(thresholds): {1, 100, inf}: \
![training_from_random_bs32_lr1e-3-clamp1](https://github.com/user-attachments/assets/0441aec7-90d8-4f4d-99a7-8d1913447824)
![training_from_random_bs32_lr1e-3-clamp100](https://github.com/user-attachments/assets/5679264f-12d6-415d-941a-c9ae25840570)
![training_from_random_bs32_lr1e-3-no_clamp](https://github.com/user-attachments/assets/54cbbff5-e951-4b01-9b7f-206455b22239)

## Conclusion:
The best configuration for this cart-pole-v1 environment is: 
    
    bs: 256
    lr:1e-3
    penalty reward: -15
    clamp value: [-1,1]

## My Question: 
Why does DRL display a sudden period of convergence and then a sudden overfitting in this experiment, unlike training computer vision models? \
-- In value-based methods, we use an aggressive operator to change the value function: we take the maximum over Q-estimates. Consequently, the action probabilities may change dramatically for an arbitrarily small change in the estimated action values if that change results in a different action having the maximal value. (from [huggingface-DRL](https://huggingface.co/learn/deep-rl-course/en/unit4/advantages-disadvantages))

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
