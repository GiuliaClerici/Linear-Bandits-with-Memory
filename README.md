# Linear-Bandits-with-Memory
This repository contains the code for Linear Bandits with Memory (LBM), to reproduce the experiments presented in a paper currently under revision.

This repository contains the following files:
- [exp.py](exp.py) contains the code to reproduce the performance of O3M, OFUL and Greedy on a rotting instance of LBM. 
- [model_selection_alpha.py](model_selection_alpha.py) contains the code to reproduce the performance of Bandit Combiner for the setting where the parameter $\gamma$ is misspecified. 
- [model_selection_m.py](model_selection_m.py) contains the code to reproduce the performance of Bandit Combiner for the setting where the parameter $m$ is misspecified.
- [greedy_subopt_benchmark.py](greedy_subopt_benchmark.py) contains the code to reproduce the performance of O3M, Policy $\pi_2$, OFUL, and Greedy for the experiment in Fig. 1 (right pane). 
- [plot.py](plot.py) contains the code to plot the results obtained from exp.py, model_selection_alpha.py, model_selection_m.py,and showed in Fig. 1 (left pane).
- [regret_rising.py](regret_rising.py) contains the code to reproduce the experiment where we compare the regret of O3M and OM-Block for several time horizons. 
- [regret_rising_plot.py](regret_rising_plot.py) contains the code to plot the results obtained from regret_rising.py for the experiment in Fig. 2 in the Supplementary Material.  

In order to run the experiments presented in Fig. 1 (left pane), run (~2-3 hours):
```python
$ python exp.py
$ python model_selection_alpha.py
$ python model_selection_m.py
$ python plot.py
```

In order to run the experiment related to Fig. 1 (right pane), run (~5 mins):
```python
$ python greedy_subopt_benchmark.py
$ python plot_greedy_subopt.py
```

To run the additional experiment presented in the Supplementary Material, run the following command (~1 week):
```python
$ python regret_rising.py
$ python regret_rising_plot.py
```
