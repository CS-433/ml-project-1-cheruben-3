# Machine Learning Project 1

## From the project introduction:
In this project, we will apply machine learning techniques to actual CERN particle accelerator data to recreate the process of “discovering” the Higgs particle.

For some background, physicists at CERN smash protons into one another at high speeds to generate even smaller particles as by-products of the collisions. Rarely, these collisions can produce a Higgs boson. Since the Higgs boson decays rapidly into other particles, scientists don’t observe it directly, but rather measure its “decay signature”, or the products that result from its decay process.

Since many decay signatures look similar, it is our job to estimate the likelihood that a given event’s signature was the result of a Higgs boson (signal) or some other process/particle (background). In practice, this means that we will be given a vector of features representing the decay signature of a collision event, and asked to predict whether this event was signal (a Higgs boson) or background (something else). To do this, we will use the binary classification techniques we have discussed in the lectures.

## Links
- Project AiCrowd page: https://www.aicrowd.com/challenges/epfl-machine-learning-higgs
- In-depth description by HiggsML: https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf