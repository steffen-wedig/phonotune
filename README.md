# Phonotune: Finetuning machine learning interatomic potentials for phonon prediction
Phonotune prepares data from the Alexandria Phonon Database for finetuning the machine learning interatomic potential MACE.
After finetuning, various utilities are available to compare the performance of models on force evaluation and phonon prediction task.

## Installation

To install the project, create a new conda environment with python 3.11.

Activate the environemnt and clone the mace github repository. Thereby you can use the cu-equivariance acceleration.


```
conda activate phonotune
git clone https://github.com/ACEsuit/mace.git
pip install ./mace

git clone https://github.com/steffen-wedig/phonotune.git
pip install ./phonotune
```
