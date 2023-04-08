from agent.agent import Agent
from tetris_gym.envs.tetris_gym import TetrisGym
from tetris_gym.utils.eval_utils import evaluate_agent

# Környezet létrehozása
env = TetrisGym(width=10, height=20)

agent = Agent(env)

print(evaluate_agent(env, agent, 100))
