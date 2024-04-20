<!--
 * @Author: WANG CHENG
 * @Date: 2024-04-20 02:20:51
 * @LastEditTime: 2024-04-20 21:35:34
-->
# RL4OCC Project

RL4OCC is a project that utilizes Deep Reinforcement Learning (DQN) to solve assembly sequence problems. The project employs both Genetic Algorithm (GA) and DQN to optimize the assembly sequence, reduce the number of collisions during assembly, and improve assembly efficiency.

## Project Structure

Below is an overview of the project's directory structure and important files:
```
rl4occ/
├── assembly.py
├── attention_q_net.py
├── data
├── data_process.py
├── dqnlearn.py
├── env.py
├── LICENSE
├── main.py
├── misc
├── misc.py
├── model
├── optimise.py
├── README.md
├── reference
├── replay_buffer.py
└── utils.py
```

## Dependencies Installation

Run the following command in the project root directory to install the required Python libraries:

```bash
pip install -r requirements.txt
```

## Usage
Running DQN Training
Execute the following command to start DQN training:
```
python main.py
```
Running Genetic Algorithm Optimization
Use the following command to run the Genetic Algorithm optimization process:
```
python optimise.py
```

## License
This project is licensed under the MIT License.
