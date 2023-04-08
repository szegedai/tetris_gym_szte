# Megerősítéses tanulás kötelező program

A feladat egy tetris ágens készítése. A játék a tetris egy egyszerűsített változata, ahol minden lépésben egy elemet kell ledobnunk.

Egy lépésben két paramétert kell beállítanunk, hogy melyik oszlopba rakjuk le az elemet, és, hogy az elem a 4 forgatási iránya közül melyikbe álljon.

<p align="center">
  <img src="docs/media/tetris_sample.gif" width=400><br/>
</p>

# Tetris

A tetris lényege, hogy a kjelzőn fentről lefele haladó 7 féle tetromino elemet úgy helyezzük el, hogy azok minél kevesebb lyukat hadjanak egymás között. Ha egy sor minden eleme lefedésre kerül, akkor az a sor eltűnik, több helyet hagyva az új elemek elhelyezésére. [Részletes szabályok](https://tetris.wiki/Tetris_Guideline).

A játékot bárki kipróbálhatja például az alábbi oldalon: [tetr.io](https://tetr.io/)


A 7 féle tetromino-t névvel ellátva az alábbi ábra tartalmatta:
<p align="center">
  <img src="docs/media/tetromino_names.png"><br/>
</p>


# Környezet

A környezetben minden lépés egy elemleejtése a tábla tetejéről. 

|   |   |
|-------------------|------------------------------|
| Action space      | <pre>MultiDiscrete([*boardWidth*  4])</pre> |
| Observation space | <pre>Dict(<br>  board: Box(0, 1, (*boardHeight*, *boardWidth*))<br>  piece: Discrete(7)<br>)</pre>|

Ahol a *boardWidth* és *boardHeight* a tábla szélességét és magásságát jelölik.

Akciók:
 - A lehetséges akciók első paraméterének annak az oszlopnak a számát kell megadnunk 0-tól indexelve, amelyikbe az adott elem legbaloldalibb kockáját szeretnénk rakni. Ha az általunk megadott oszlopba már nem lehetséges elemet rakni (például egy "Z" elemet akarunk az utolsó oszlopba rakni), akkor a legjobboldalibb még lehetséges oszlpba fogja ranki a rendszer.
 - A második paraméter az elforgatás, ami minden elem esetén 4 irány lehet, mégha bizonyos elemeknél ezek át is fednek.

Megfigyelések:
 - A megfigyeléseket egy Dictionary-ben kapjuk meg, aminek két kulcsa van: a *board* és a *piece*.
 - A *board* egy 2 dimenziós tömbben tartalmazza a jelenlegi táblát, ami 0, ha a mező még üres és 1, ha már tettünk oda elemet.
 - A *piece* egy egész szám, ami a lerakandó tetromino azonosítóját tartalmazza, a az értéke 0-tól az elemek száma - 1 tart. Mivel az elemek száma környezetenként eltérhez, ezért egy adott tertominohoz külön környezetekben külön azonosítók tartozhatnak.
 - Mivel a környezet egy Dictionary-ben tér vissza az elemekkel a *stable-baselines3* esetén alapértelmezettként a [MultiInputPolicy](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#stable_baselines3.ppo.MultiInputPolicy) használata ajánlott. (Lásd: példa kódok)

 A környezet a *TetrisGym* osztállyal példányosítható, az alábbi módon:
 ```python
 from tetris_gym.envs.tetris_gym import TetrisGym

 env = TetrisGym(width=6, height=14, pieces=["O", "I", "J", "L"])
 ```

 Ahol a 3 játékmenet befolyásoló paraméter az alábbiak:
  - *width*: a tábla szélessége
  - *height*: a tábla magassága
  - *pieces*: A játékban használható tetrominok neve, lehetséges értékek: *O, I, J, L, T, S, Z*

# Pontszámítás

Minden lerakott elem 1 pontot ér. Ha egy lerakott elem hatására eltűnnek sorok, akkor ez az eltűnt sorok száma a négyzeten szorozva a tábla szélességével további jutalmat jelent.

$$ score = 1 + clearedLines^2 * boardWidth $$

Ahol a *clearedLines* az eltűntetett sorok számát a *boardWidth* pedig a pálya méretét jelenti.
Ha mondjuk egy hagyományos 10 széles pályát egy lépéssel 3 sort eltüntetünk, akkor $1 + 3^2 * 10 = 91$ pontot kapunk.

Az alapértelmezett jutalom megegyezik a pontokkal.

# Telepítés és futtatás

A rendszer egyaránt használható google colabon és hagyományos számítógépeken. A környezet egy átlagos laptop processzorán is kényelmesen futtatható.

Példa [colab notebook](https://colab.research.google.com/drive/1ql97tMmdBu_349S6cxWMBxbeMUcoPr7v?usp=sharing).

Az alábbi útmutatóban [conda](https://docs.conda.io/en/latest/) virtuális környezetet fogunk használni.

Conda környezet létrehozása:

```bash
conda create -n tetris_gym python=3.10
conda activate tetris_gym
```

Rendszer letöltése és a csomagok telepítése:

```bash
git clone https://github.com/szegedai/tetris_gym_szte.git

cd tetris_gym_szte

pip install setuptools==65.5.0

pip install -r requirements.txt
```

Példakód kipróbálása:

```bash
python example.py
```

# Kiértékelés és követelmények

A végleges környezet a hagyományos Tetris játékhoz hasonlóan 10 széles és 20 magas táblát használ, a környezetben mind a 7 féle tetromino-t használjuk.

Az ágenst a [agent/agent.py](agent/agent.py) fájlban kell megvalósítani. Ezt fogja meghívni a végleges kiértékelő rendszer.

Az ágensben az *act* metódust kell módosítani, ami a környezetből kapott megfigyelés alapján visszaadja a következő lépést.

Ezzen felül a konstruktorban lehetőség van az ágensünk inicializálására, például egy korábban tanult modell betöltésére. Illetve, amennyiben használtunk *wrapper*-eket a környezet modosításához azokat is lehtőségünk van itt létrehozni.

Egy példa ágens, ami egy betanított Stable Baselines3 modellt használ az alábbi módon nézz ki:
  
  ```python
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
  ```

## Felhasználható csomagok

Természetesen a Stable Baselines3 használata nem kötelező, lehetőség van tetszőleges modell, illetve egyénileg írt kód használatára is.

A kiértékelő rendszerben az alábbi csomagok vannak telepítve.

Új csomagok telepíthetők, ha erre van igényetek kérlek jelezzétek a kötelező programhoz létrehozott coospace forumon.

## Ranglista

A ranglista és a feltöltés az alábbi oldalon érhető el:

[https://chatbot-rgai3.inf.u-szeged.hu/rl/](https://chatbot-rgai3.inf.u-szeged.hu/rl/)

## Feltöltés

Az elkészült kódokat fel kell tötenetek HuggingFace-re. Majd, ha úgyérzitek, hogy minden rendben van, akkor a [ranglista oldalán](https://chatbot-rgai3.inf.u-szeged.hu/rl/upload/) tudjátok elindítani a hivatalos kiértékelést. Ehhez meg kell adnotok a HuggingFace repository nevét, ahova feltöltöttétek a kódotokat és a modelleket, a neptun azonosítótokat és egy megjelenítéshez használni kívánt nevet.

A HuggingFace repository-ba mindent fel kell tölteni, ami szükséges a kód futtatásához. Ez magában foglalja a kódokat és a szükséges modelleket. Az *agent.py*-nak a repository gyökérkönyvtárában kell lennie. Példát erre az alábbi repository-ban találtok: [szterlcourse/tetris_example](https://huggingface.co/szterlcourse/tetris_example/tree/main)

### Példa

Az alábbi [notebook](https://colab.research.google.com/drive/1iO9J_VzrtSIVcjC5a3q-9pk9UtfHX1SZ?usp=sharing), illetve a lenti parancsok megmutatják hogyan tudtok betanítani, leellenőrizni és feltöteni a Hugging Face-re egy ágenst.

A modellt betíníthatod a [trani.py](train.py) fájl segítségével, ez létre fog hozni egy modellt az *agent* mappában.
```bash
python train.py
```

Fontos, hogyha tanításkor modosítottál a tanuló algoritmuson, a megfigyeléseken... vagy csak egyéni szabályokat szeretnél írni, akkor már a korábban említett [agent/agent.py](agent/agent.py) fájlt kell ehhez módosítanod.

A kész ágenst az [evaluate.py](evaluate.py) fájl segítségével ellenőrizheted.
```bash
python evaluate.py
```

A modellt feltöltheted a HuggingFace-re a [upload.py](upload.py) fájl segítségével.

Ehhez viszont először meg kell adnod a fájlban a létrehozni (vagy felülírni) kívánt repository nevét. Illetve a Hugging Face tokenedet. Ezt az alábbi helyen tudod létrehozni a Hugging Face-en belül: [Settings/Access Tokens](https://huggingface.co/settings/token)

```python
# Ezt át kell írni a saját felhasználónevedre és az általad választott repó nevére
# Pl.: "szterlcourse/my_agent"
repo_id = ""

# Ide be kell írni a saját tokenedet, amit a Hugging Face oldalán tudsz létrehozni (https://huggingface.co/settings/token)
token = ""
```

Ha ezek megvannak, akkor a fájl futtatásával feltöltheted a Hugging Face-re a kódot és a modelleket.
```bash
python evaluate.py
```

A feltöltést kézzel is megteheted, de ezekben van arra példa, hogy hogyan lehet kódból létrehozni egy Hugging Face repository-t és feltölteni bele a kódot és a modelleket.

## Hibák megjelentetése

Az utolsó feltöltés log-ját a neptun kódotok segítségébvel az alábbi link módosításával tudjátok megnézni:

[https://chatbot-rgai3.inf.u-szeged.hu/rl/log/\<NEPTUNKOD\>/](https://chatbot-rgai3.inf.u-szeged.hu/rl/log/NEPTUNKOD/)

Fontos, hogy a záró / szükséges.

# Követelmények
A kötelező programért szerezhető 30 pont begyűjtéséhez fel kell töltened egy rendszert, ami a szerveren történő kiértékeléskor legalább 40 score-t ér el.

A legjobb 5 felöltő mentesül az elméleti zh alól.

A további helyezések extra pluszpontokat érnek, amiknek a pontos szabályait a későbbiekben részletezzük.

## Köszönenyílvánítás

Köszönet  [Viet Nguyen](https://github.com/uvipen)-nek a tetris környezet alapjaiért.
