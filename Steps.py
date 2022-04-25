from Downloads.DigDugProject.Actions import Actions

def start_game(frame_ratio):
    return [
        {"wait": int(300/frame_ratio), "actions": [Actions.COIN]},
        {"wait": int(60/frame_ratio), "actions": [Actions.START]}]

