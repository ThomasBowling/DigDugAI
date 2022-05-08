from MAMEToolkit.emulator import Emulator
from MAMEToolkit.emulator import Address
from Actions import Actions
from Steps import start_game
import re

def setup_memory_addresses():
    return {
        "score": Address('0x8414', 'u32'),
        "round": Address('0x840D', 'u8'),
        "lives": Address('0x840A', 'u8'),
        "coin": Address('0x85A5', 'u8'),
        "gameActive": Address('0x8400', 'u8')
    }   

def index_to_action(action):
    return {
        0: [Actions.LEFT],
        1: [Actions.UP],
        2: [Actions.RIGHT],
        3: [Actions.DOWN],
        4: [Actions.ATTACK]
    }[action]

#Hold attack?
    
class Environment(object):

    def __init__(self, env_id, roms_path, frame_ratio=6, render=True):
        self.frame_ratio = frame_ratio
        self.emu = Emulator(env_id, roms_path, "digdug", setup_memory_addresses(), frame_ratio=frame_ratio, render=render)
        self.Dims = self.emu.screenDims
        self.started = False
        self.game_done = False
        self.inital_score = 0
        self.inital_round = 1
        self.inital_lives = 2

#start
    def run_steps(self, steps):
        for step in steps:
            for i in range(step["wait"]):
                self.emu.step([])
            self.emu.step([action.value for action in step["actions"]])
            
    def start(self):
        self.run_steps(start_game(self.frame_ratio))
        self.started = True
    
    def check_done(self, data, reward):
        if data["gameActive"] == 0:
            reward = reward-1
            converted_score = self.convert_score(data["score"])
            print("score: " + str(converted_score))
            self.game_done = True
        return data, reward
#new game
    def new_game(self):
        self.run_steps(start_game(self.frame_ratio))
        self.inital_score = 0
        self.inital_round = 1
        self.inital_lives = 2
        self.game_done = False
        
    def convert_score(self, score):
        fill = (len(hex(score))-2)*4;
        bin_val = str(bin(score))[2:].zfill(fill)
        bin_arr = (re.findall('.{1,4}', bin_val))
        converted_score = "";
        for digit in bin_arr:
            digit = int(digit, 2)
            converted_score += str(digit)
        return int(converted_score)
    
    
    def get_reward(self, score, round_val, lives):
        reward = ((score-self.inital_score)/10) + ((round_val-self.inital_round)) - ((self.inital_lives-lives))
        self.inital_score = score
        self.inital_round = round_val
        self.inital_lives = lives
        return reward
        
#step

    def step(self, action):
        if self.started:
            if not self.game_done:
                actions = []
                actions += index_to_action(action)
                data = self.emu.step([action.value for action in actions])
                converted_score = self.convert_score(data["score"])
                reward = self.get_reward(converted_score, data["round"], data["lives"])                
                data, reward = self.check_done(data, reward)
                return data["frame"]/255, self.game_done, reward, data["round"]
            
#close
    def close(self):
        self.emu.close()
    