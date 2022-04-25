from enum import Enum
from MAMEToolkit.emulator import Action

class Actions(Enum):
    #Start
    COIN = Action(':IN0H', 'Coin 1')
    START = Action(':IN0L', '1 Player Start')

    # Movement
    UP = Action(':IN1L', 'P1 Up')
    DOWN = Action(':IN1L', 'P1 Down')
    LEFT = Action(':IN1L', 'P1 Left')
    RIGHT = Action(':IN1L', 'P1 Right')
    
    #Attack
    ATTACK = Action(':IN0L', 'P1 Button 1')