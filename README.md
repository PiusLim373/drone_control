# drone_control
This is a machine learning based drone controlling project for NUS MSc in Robotics module ME5418: Machine Learning in Robotics.

## Setup
### 1. Clone the repo
```
git clone https://github.com/PiusLim373/drone_control.git
```
### 2. Setup using Conda
```
cd drone_control
conda env create -f requirement.yml
conda activate me5418-drone-control
```
This will create a conda environment me5418-drone-control with necessary packages installed to run the project.

## Running the Gym Environment Demo (Assignment 1)
Assignment 1 is about showing a working version of the gym environment that will allow subsequent program to call and step through the environment with specific action and get some state data in return.
``` 
python gym_demo.py
```
This demo script will create an environment in demo mode and run for 140steps for visualization. There are 4 checkpoints in this demo.

:warning: You will need to press ECS key to end the rendering at the end of each checkpoint so that the demo will continue.

#### Checkpoint 1
First 25steps, spawn the quadcopter and wait for it to reach a stable stationary state.
![](asset/docs/drone_stationary.png)

#### Checkpoint 2
For the next 50steps, activate all 4 rotors and the quadcopter will take off.
![](asset/docs/drone_tookoff.png)

#### Checkpoint 3
For the next 50steps, activate the diagonal rotor 1 and rotor 3 and the quadcopter will rotate perform yaw rotation.
![](asset/docs/drone_yaw_rotation.png)

#### Checkpoint 4
For the next 15steps, activate the back rotor 3 and rotor 4 and the quadcopter will rotate perform pitch rotation.
![](asset/docs/drone_pitch_rotation.png)

### 3. Unit Tests
```
python gym_unittests.py
```


## Running the Neural Network Demo (Assignment 2) 
Stay tuned, coming soon!
## Running the Learning Agent Demo (Assignment 3) 
Stay tuned, coming soon!