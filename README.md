# dice_rl_sepsis
This repository includes the code for the publication "Evaluating AI-based Sepsis Treatments via Tabular and Continuous Stationary Distribution Correction Estimation".

## Install

MacOS
python 3.11.9

Create a virtual environment:

```
cd <venvs_location>
python3 -m venv <venv_name>
source <venv_name>/bin/activate
```

Install all the necessary packages:

```
pip install tfp-nightly
pip install tf-agents-nightly
pip install keras
```

```
pip install matplotlib
pip install pandas
pip install tqdm
pip install stable-baselines3
pip install scikit-learn
pip install openpyxl
pip install pyarrow
```

In `<venv_name>/lib/python3.11/site-packages/tf_agents/networks/utils.py`,
replace `'/'` by `'_'`.


## Structure

### Generic Algorithm Code

- `dice_rl`:
fork of the original repository [`dice_rl`](https://github.com/google-research/dice_rl.git)

- `dice_rl_TU_Vienna`:
re-implementation of all Dice-algorithms and related base calsses for tools like analytical solvers, plotting, dataset conversion, etc.

### Application Specific `dice_rl`-Code

- `plugins`:
outsource code for your custom application, so you don't have everything in the `.ipynb`-notebooks and `.py`-scripts

- `.ipynb`-notebooks and `.py`-scripts are run on the highest hierachical level.

### Application Specific Dependencies

- `medical_rl`:
preprocessors and base classes for medical reinforcement learning applications

- `sepsis_amsterdam`:
code for sepsis treatement via reinforcement learning using [AmsterdamUMCdb](https://amsterdammedicaldatascience.nl/amsterdamumcdb/)
