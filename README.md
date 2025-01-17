# NSDE

Code for Neural Stochastic Differential Equation. Implements SDEs with multiplicative and additive noise, on OU path dataset

Implemented from
- Oh, Y., Lim, D., & Kim, S. (2024). Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data. arXiv preprint arXiv:2402.14989.
- Xuanqing Liu, Tesi Xiao, Si Si, Qin Cao, Sanjiv Kumar, and Cho-Jui Hsieh. Neural sde: Stabilizing neural ode
networks with stochastic noise. arXiv preprint arXiv:1906.02355, 2019.

## Running the code

- Clone the repo and navigate to the project directory
- For conda users, create and activate a conda environment
  ```bash
  conda env create -f environment.yml
  conda activate NSDE_Env
  ```
- For pip users, install requirements directly (or by using pyenv first)
  ```bash
  pip install requirements.txt
  ```
- Run the script
  ```bash
  python main.py
  ```

## Additional Information

- Model configs stored in /model_config.yaml
- OU path configs stored in /datasets/OU_config.yaml
- Plots are stored in /plots/
