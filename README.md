## Connect-4 with Q-learning
An investigation into the utility of reinforcement
Q-learning through versions of "Connect-4". Built 
for limit testing Q-learning. (AI performs poorly on 
traditional board size!)
### How to Play
Clone/copy connect_4.py (no requirement installation needed).
specify the board width, height, how many connections 
are needed for a victory, and how many games to train on in the command line.

python connect_4.py width height connect training_games

"python connect_4" will be the default game with 100,000 training games.

I recommend deleting the agents directory after use as it takes up a lot of space.

### Comments
The connect 4 state space is sufficiently large that the agent tends to be forced
into making uniformed choices within a few moves. A deep q learning approach would fare 
much better, and in this implementation the q dictionary will take up far to much space before
becoming useful.

Q-learning still performs well on the smaller spaces, in particular
its performance in the "gravity obeying tic-tac-toe" with 10,000 games played:

python connect_4.py 3 3 3 10000

is already analogous to minimax. printing 'maximum' in 'highest_future_reward'
will indicate whether the agent is making informed choices, on defaulting to 0 for the unknown.