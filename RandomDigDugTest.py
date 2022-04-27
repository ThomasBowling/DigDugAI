import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from Downloads.DigDugProject.Environment import Environment

no_actions = 5

def ann_model():
    inputs = tf.keras.layers.Input(shape=(env.Dims["height"], env.Dims["width"], 3,))
    layer1 = tf.keras.layers.Conv2D(32, (8, 8), strides=4, activation="relu")(inputs)
    layer2 = tf.keras.layers.Conv2D(64, (4, 4), strides=2, activation="relu")(layer1)
    layer3 = tf.keras.layers.Conv2D(64, (2, 2), strides=2, activation="relu")(layer2)
    
    val_stream, adv_stream  = tf.split(layer3, 2, 3)

    val_stream = tf.keras.layers.Flatten()(val_stream)
    val = tf.keras.layers.Dense(1, activation="linear")(val_stream)
    
    adv_stream = tf.keras.layers.Flatten()(adv_stream)
    #no of actions
    adv = tf.keras.layers.Dense(no_actions, activation="linear")(adv_stream)

    q_vals = val + (adv - tf.reduce_mean(adv, axis=1, keepdims=True))

    # Build model
    model = tf.keras.Model(inputs, q_vals)

    return model

roms_path = "roms/"  # Replace this with the path to your ROMs
env = Environment("env1", roms_path)
env.start()
agent = ann_model()
game = 0;
action = 0

while game < 100:
    
    state, game_done = env.step(action)
    pred = agent.predict(state.reshape((-1, env.Dims["height"], env.Dims["width"], 3)))[0]
    action = pred.argmax()
    # action = random.randint(0, no_actions-1)
    # frames, game_done = env.step(action)
    
    # state = state.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work
    # shape = (env.Dims["width"], env.Dims["height"], 3);
    # state = state.reshape((*shape, 1))
    # plt.imshow(np.array(np.squeeze(state)), cmap='gray', vmin = 0, vmax = 1)
    # plt.show()
    if game_done:
        game += 1;
        print("game: " + str(game))
        env.new_game()
env.close();