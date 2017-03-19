Stochastic Rank-1 Bandits
====

rank1bandits implements algorithms Rank1Elim [1] and Rank1ElimKL [2]

# How to Use
The algs folder contains the code for the two algorithms in the respective python files. The envs folder contains code to simulate different environments. As of this writeup, two environments have been added - the Spike environment containing a single attractive item and position, and the Rating environment that reads the attraction and examination probabilities from csv files.

An example of how to run the code can be found in main.py.

# Contributors
Branislav Kveton, Csaba Szepesvári, Claire Vernade, Zheng Wen.

# References
[1] Sumeet Katariya, Branislav Kveton, Csaba Szepesvári, Claire Vernade, Zheng Wen. Stochastic Rank-1 Bandits. In Artificial Intelligence and Statistics, Fort Lauderdale, USA, 2017.

[2] Sumeet Katariya, Branislav Kveton, Csaba Szepesvári, Claire Vernade, Zheng Wen. Bernoulli Rank-1 Bandits for Click Feedback. Arxiv.
