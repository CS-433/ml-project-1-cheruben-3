# Machine Learning Project 1: Discovering the Higgs boson using machine learning

## From the project introduction
In this project, we will apply machine learning techniques to actual CERN particle accelerator data to recreate the process of “discovering” the Higgs particle.

For some background, physicists at CERN smash protons into one another at high speeds to generate even smaller particles as by-products of the collisions. Rarely, these collisions can produce a Higgs boson. Since the Higgs boson decays rapidly into other particles, scientists don’t observe it directly, but rather measure its “decay signature”, or the products that result from its decay process.

Since many decay signatures look similar, it is our job to estimate the likelihood that a given event’s signature was the result of a Higgs boson (signal) or some other process/particle (background). In practice, this means that we will be given a vector of features representing the decay signature of a collision event, and asked to predict whether this event was signal (a Higgs boson) or background (something else). To do this, we will use the binary classification techniques we have discussed in the lectures.

## About the dataset
The different models proposed are based on machine learning binary classification techniques, trained and tested with actual CERN particle accelerator data. The training set includes 250,000 events, which around the 34.27% are ’signals’, and the test set includes around 568,238 events. The amount of missing data in both sets is approximately 21%, which will need to be handled. 

The number of features provided for each vector are 30, from which 17 are raw quantities (primitives) about the collision measured by the detector and the rest (13) are quantities computed (derived) from the primitive features. Only one column is discrete (<i> PRI_jet_num </i>), which will have to be addressed as well.

## Structure of the repository
    .
    ├── data                    # Data files
    ├── documents               # Documention given about project 1
    ├── experimenting           # Data analysis, tests and notebook of project
    ├── visualizations          # Plots based on observation of the data and 4-fold cross-validation
    ├── .gitignore              # Configuration settings
    ├── README.md               # Description of repository
    ├── helpers.py              # Data analysis, tests and notebook of project
    ├── implementations.py      # Plots based on observation of the data and 4-fold cross-validation
    ├── metrics.py              # Data analysis, tests and notebook of project
    └── run.py                  # Code with the best submission for the project

### Data
    .
    ├── data                   
    │   └── .gitignore          # Git ignores files here
    └── ...

> This is the folder where all the data files are stored. As it's written in the `.gitignore`, the data files have to be downloaded from <i> https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/dataset_files </i> and unzipped here, so it can be called by the other files. Here it will also be stored the submission files created.

### Documents
    .
    ├── ...
    ├── documents                   
    │   ├── documentation_v1.8.pdf          # Description of the Higgs boson dataset
    │   ├── example_project_1_feedback.pdf  # Usual comments for improvement regarding project 1
    │   └── project1_description.pdf        # Description of the ML project 1
    └── ...

> In this folder is collected all the documentation given for carrying out the ML project 1. The main file is `project1_description.pdf`, where the requirements to be met, as well as the grading criteria, steps to follow in the project and general information about the task is described. The `example_project_1_feedback.pdf` is a collection of the usual details that can be improved for project 2 regarding plots, report, methodology followed, etc. and the `documentation_v1.8.pdf` gives the background for the Higgs boson experiment and explains the physics behinds it along with the features of the dataset.

### Experimenting
    .
    ├── ...
    ├── experimenting                   
    │   ├── analysis_ana.ipynb              # First analysis of PRI_tau/lep/met features
    │   ├── analysis_anton.ipynb            # First analysis of all features
    │   ├── analysis_lars.ipynb             # First analysis of PRI_jet features
    │   ├── feature expansion tests.ipynb   # Testing of data expansion with polynomial features
    │   └── run.ipynb                       # Jupyter notebook of the whole project 1
    └── ...

> This folder contains the first analysis performed on the dataset given, so some information about the data is learnt, as well as the testing of the feature expansing with polynomial features and the jupyter notebook of the whole project 1.

#### Analysis
These files were the first steps taken regarding the extraction of information about the features: plotting their distributions with histograms (which are stored in the subfolder `histograms`), looking for correlations between them (figures about the PRI_jet features are stored in the subfolder `jet plots`), the use of said features in the experiments... The knowledge extracted leads the way for the next steps.

#### Feature expansion tests
In order to gain accuracy for the model without falling into overfitting, some polynomial feature expansion is performed and tested with logistic regression to know if it is worth it by comparing the mean errors of the model with and without the expansion.

#### run.ipynb
Jupyter notebook of the project where all the steps taken are explained: loading and preprocessing of the data, where the data cleaning is performed; training and analysis, where the 4-fold cross-validition for logistic and ridge regression optimal coefficient is computed (figures about the lambda tested in both methods are store in the subfolder `cross-validation`), along with the testing and evaluation (ROC curves, accuracy, f1-score) of several models; benchmarking several algorithms and inference of a certain model. It also details the train of thought followed.

### Visualizations
    .
    ├── ...
    ├── visualzations                   
    │   ├── cross-validations               # 4-fold cross-validations plots
    │   ├── histograms                      # Feature distrubutions with histograms plots
    │   └── jet plots                       # PRI_jet feature plots
    └── ...

> This folder contains subfolders with most of the figures created in the project, as it has been already mentioned. The folder `cross-validation` contains the relation between the lambdas tested and the errors obtained with the regularized logistic regression and ridge regression models; the folder `histograms` contains feature distributions plotted as histograms and the folder `jet plots` contains the joint-plots of the PRI_jet features, where some correlation can be spotted.

### Other files
#### helpers.py
> File with useful functions that are called in the files run.py and run.ipynb to keep the project cleaned and organised. They are used for the loading and preprocessing of the data, as well as for obtaining the best lambda for the methods tested.

#### implementations.py
> File with the ML methods implemented that are required: mean_squared_error_gd, mean_squared_error_sgd, least_squares, ridge_regression, logistic_regression and reg_logistic_regression. Also, the reg_logistic_regression_AGDR is implemented, as it was thought that said method could a better model for the dataset. 

#### metrics.py
> File with the functions for computing the losses of the methods implemented in `implementations.py` and for calculating the confusion matrix statistics, which returns quite useful information (for instance, the f1 score and the accuracy of a method for a binary classification problem).

#### run.py
> File with information similar to run.ipynb, but focus on replicating the best model obtained in the project and submitted.


## Links
- Overleaf: https://www.overleaf.com/read/yzksyvdpcgcg
- Project AiCrowd page: https://www.aicrowd.com/challenges/epfl-machine-learning-higgs
- In-depth description by HiggsML: https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf

## Credits

Project carried out by Lars Quaedvlieg, Anton Pirhonen and Ana Serrano Leiva. 

Group CheRuben.