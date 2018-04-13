# rocketlander_deep_q_learning

RocketLander gym (OpenAI) solution through Double Deep Q Network

Quentin BB
Alexis M
Paul W

Rocket and Lunar Landers solutions.

To run our agents you need to install pybox2d and use the gym toolkit with the
extra environment RocketLander.

https://github.com/EmbersArc/gym

The agents we implented are in the folder `agents`, and the most important 
one is the `DQL_agent.py` agent implementing our DeepQ and Double-DeepQ 
learning algorithms.

The best agent we trained on LunarLander can be tested with the following command:

python3 main.py --env LunarLander --display --niter 400 --load_model lunar_1000G_DoubleDQN --ngames 10
