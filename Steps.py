from Actions import Actions

def start_game(frame_ratio):
    return [
        {"wait": int(180/frame_ratio), "actions": [Actions.COIN]},
        {"wait": int(60/frame_ratio), "actions": [Actions.START]},
        {"wait": int(534/frame_ratio), "actions": [Actions.START]}]

