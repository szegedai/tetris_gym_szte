from stable_baselines3 import A2C
from tetris_gym.wrappers.observation import ExtendedObservationWrapper

class Agent:
    """
    A kötelező programként beadandó ágens leírása.
    """

    def __init__(self, env) -> None:
        """
        A konsztruktorban van lehetőség például a modell betöltésére
        vagy a környezet wrapper-ekkel való kiterjesztésére.
        """
        
        self.model = A2C.load("agent/model_20x10")
        
        # A környezetet kiterjeszthetjük wrapper-ek segítségével.
        # Ha tanításkor modosítottuk a megfigyeléseket,
        # akkor azt a módosítást kiértékeléskor is meg kell adnunk.
        self.observation_wrapper = ExtendedObservationWrapper(env)

    def act(self, observation):
        """
        A megfigyelés alapján visszaadja a következő lépést.
        Ez a függvény fogja megadni az ágens működését.
        """

        # Ha tanításkor modosítottuk a megfigyeléseket,
        # akkor azt a módosítást kiértékeléskor is meg kell adnunk.
        extended_obsetvation = self.observation_wrapper.observation(observation)

        return self.model.predict(extended_obsetvation, deterministic=True)
    