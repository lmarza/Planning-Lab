# Planning-Lab
Code for the Planning lab of Planning and Automated Reasoning course of the MSc in Artificial Intelligence 2022/2023 of the University of Verona

## Setup 
1. Download [Anaconda](https://www.anaconda.com/distribution/#download-section) for your System.

2.  Install Anaconda
	- On Linux/Mac 
		- Use *sh Anacaonda{version}.sh* to install.
		- Add it to the PATH during installation if you’re ok with it:
			- First *export PATH=~/anaconda3/bin:$PATH*
			- Then *source ~/.bashrc*
		- *sudo apt-get install git* (may be required).
	- On Windows
		- Double click the installer to launch.
		- *NB: during the installation, ensure to install "Anaconda Prompt" and use it for the other steps.*

3.  Set-Up conda environment:
	- *git clone https://github.com/LM095/Planning-Lab*
	- *conda env create -f tools/ai-lab-env.yml*

## Using the Notebook
To start the environment and work on your assignments, navigate to the downloaded folder root *(AI-Lab)* and run:
```
conda activate ai-lab
jupyter notebook
```
The last command will open your browser for you to start working. To open the tutorial navigate with your browser to the current lesson notebook (*Lesson_\*/lesson_\*_problem.ipynb*).

## Authors
*  **Luca Marzari** - luca.marzari@univr.it
*  **Alessandro Farinelli** - alessandro.farinelli@univr.it

## Acknowledgments
Environments are based on OpenAI Gym: https://github.com/openai/gym
