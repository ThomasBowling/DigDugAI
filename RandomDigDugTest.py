import random
import matplotlib.pyplot as plt
import numpy as np
from Downloads.DigDugProject.Environment import Environment

roms_path = "roms/"  # Replace this with the path to your ROMs
env = Environment("env1", roms_path)
env.start()
game = 0;
while game < 100:
    action = random.randint(0, 5)
    frames, game_done = env.step(action)
    frames = frames.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work
    shape = (288,224,3);
    frames = frames.reshape((*shape, 1))
    plt.imshow(np.array(np.squeeze(frames)), cmap='gray')
    plt.show()
    if game_done:
        game += 1;
        env.new_game()
env.close();
