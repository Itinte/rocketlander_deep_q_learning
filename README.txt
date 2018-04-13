This folder contains the code for the AML project of:
Quentin Boutoille-Blois
Alexis Mathey
Paul Wambergue

We worked on Rocket and Lunar Landers.

To run our agents you need to install pybox2d and use the gym toolkit with the
extra environment RocketLander.

https://github.com/EmbersArc/gym

The agents we implented are in the folder `agents`, and the most important 
one is the `DQL_agent.py` agent implementing our DeepQ and Double-DeepQ 
learning algorithms.

The best agent we trained on LunarLander can be tested with the following command:

python3 main.py --env LunarLander --display --niter 400 --load_model lunar_1000G_DoubleDQN --ngames 10