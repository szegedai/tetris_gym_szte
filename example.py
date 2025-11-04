from stable_baselines3 import A2C
from tetris_gym.envs.tetris_gym import TetrisGym
from tetris_gym.wrappers.observation import ExtendedObservationWrapper
from tetris_gym.utils.eval_utils import evaluate, create_videos

# Környezet létrehozása
polyominos = ["1", "2", "3_I", "3_L"]
env = TetrisGym(width=6, height=14, pieces=polyominos)

# A megfigyelések kiterjesztése a tábla alapján számolt új jellemzők segítségével.
env = ExtendedObservationWrapper(env)

# Modell létrehozása
model = A2C('MultiInputPolicy',  env, verbose=1, seed=42)

print(model.policy)

# Tanulás
model.learn(total_timesteps=40000)

# Model kimentése
model.save("models/model_6-14_HO_40k")

# Kiértékelés 10 véletlen környezetben
score = evaluate(env, model, 10)
print(f"Score: {score}")

# Videók készítése
create_videos(env, model)
