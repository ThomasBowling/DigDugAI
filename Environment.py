from MAMEToolkit.emulator import Emulator
from MAMEToolkit.emulator import Address
from Downloads.DigDugProject.Actions import Actions
from Downloads.DigDugProject.Steps import start_game

def setup_memory_addresses():
    return {
        "score": Address('0x8414', 'u32'),
        "round": Address('0x840D', 'u8'),
        "lives": Address('0x840A', 'u8'),
        "coin": Address('0x85A5', 'u8')
    }   

def index_to_action(action):
    return {
        0: [Actions.LEFT],
        1: [Actions.UP],
        2: [Actions.RIGHT],
        3: [Actions.DOWN],
        4: [Actions.ATTACK],
        5: [],
        6: [Actions.COIN],
        7: [Actions.START]
    }[action]

#Hold attack?
    
class Environment(object):

    def __init__(self, env_id, roms_path, frame_ratio=3, render=True):
        self.frame_ratio = frame_ratio
        self.emu = Emulator(env_id, roms_path, "digdug", setup_memory_addresses(), frame_ratio=frame_ratio, render=render)
        print(self.emu.screenDims)
        self.started = False
        self.game_done = False

#start
    def run_steps(self, steps):
        for step in steps:
            for i in range(step["wait"]):
                self.emu.step([])
            self.emu.step([action.value for action in step["actions"]])
            
    def start(self):
        self.run_steps(start_game(self.frame_ratio))
        self.started = True

    def wait_for_game_start(self):
        data = self.emu.step([])
        while data["round"] != 1:
            data = self.emu.step([])
        return data["frame"]
    
    def check_done(self, data):
        if data["lives"] == 0:
            self.game_done = True
        return data
#new game
    def new_game(self):
        self.run_steps(start_game(self.frame_ratio))
        self.game_done = False
#step

    def step(self, action):
        if self.started:
            if not self.game_done:
                actions = []
                actions += index_to_action(action)
                data = self.emu.step([action.value for action in actions])
                # print(data["score"])
                # print(data["round"])
                # print(data["lives"])
                data = self.check_done(data)
                return data["frame"], self.game_done
#close
    def close(self):
        self.emu.close()
    