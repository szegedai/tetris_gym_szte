from stable_baselines3 import A2C
from tetris_gym.envs.tetris_gym import TetrisGym
from tetris_gym.wrappers.observation import ExtendedObservationWrapper
from tetris_gym.utils.eval_utils import evaluate, create_videos

# Környezet létrehozása
env = TetrisGym(width=10, height=20)

# A megfigyelések kiterjesztése a tábla alapján számolt új jellemzők segítségével.
env = ExtendedObservationWrapper(env)

# Modell létrehozása
model = A2C('MultiInputPolicy',  env, verbose=1, seed=42)

print(model.policy)

# Tanulás
model.learn(total_timesteps=100000)

# Model kimentése
model.save("agent/model_20x10")

# Kiértékelés 10 véletlen környezetben
score = evaluate(env, model, 10)
print("Score: {}".format(score))

# Videók készítése
create_videos(env, model)