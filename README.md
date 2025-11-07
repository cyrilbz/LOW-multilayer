# Installation procedure
Get the code by cloning the repository 
```
git clone git@github.com:cyrilbz/LOW-multilayer.git
```
or getting it as a zip file (on the repository main page click on "<> Code" and then "Download ZIP").

Install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

In a terminal (or Anaconda prompt for Windows users)
```
conda create -n LOW -c conda-forge
conda activate LOW
conda install scipy matplotlib
```

# Usage
(In a terminal or Anaconda prompt)
Activate your environement 
```
conda activate LOW
```
Go into the ```code/``` directory (important as some I/O paths are written as relative paths), then use
```
python LO_multilayer.py
```
for a first run (around 30 seconds), and then to plot a few results
```
python post_process.py
```
Then you can play with some parameters and options in ```functions.py```.
