from stable_baselines3 import A2C
from tetris_gym.wrappers.observation import ExtendedObservationWrapper

class Agent:

    def __init__(self, env) -> None:
        self.model = A2C.load("agent/model_20x10")
        
        self.observation_wrapper = ExtendedObservationWrapper(env)

    def act(self, observation):

        # Ha tanításkor modosítottuk a megfigyeléseket, akkor azt a módosítást kiértékeléskor is meg kell adnunk.
        extended_obsetvation = self.observation_wrapper.observation(observation)

        return self.model.predict(extended_obsetvation, deterministic=True)
    