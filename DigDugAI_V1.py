import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from Downloads.DigDugProject.Environment import Environment

no_actions = 5

min_epsilon = 0.1 # force 0 when testing(open CV)
gamma = 0.99
decay = 0.000002 
alpha = 0.001
TENSORBOARD_DIR = 'logs/tensorboard/'

opt = tf.optimizers.Adam(alpha)

def get_epsilon(step):
    eps = min_epsilon + (1 - min_epsilon) * np.exp(-decay * step)
    return eps

def loss_with_grads(state, action, target, model):
    with tf.GradientTape(persistent=True) as tape:
        mask = tf.one_hot(action, no_actions, dtype=tf.float32)
        
        pred = model(state.reshape((-1, env.Dims["width"], env.Dims["height"], 3)))[0]
        
        loss = tf.reduce_mean(tf.reduce_sum(mask * tf.square(pred-target), axis=1))
    
    grads = tape.gradient(loss,model.trainable_weights)
    
    return loss, grads

def get_action(step, state):
    eps = min_epsilon + (1 - min_epsilon) * np.exp(-decay * step)
    # With chance epsilon, take a random action
    if np.random.rand(1) < eps:
        return np.random.randint(0, no_actions)
    
    pred = agent.predict(state.reshape((-1, env.Dims["width"], env.Dims["height"], 3)))[0]
    return pred.argmax()

def ann_model():
    inputs = tf.keras.layers.Input(shape=(env.Dims["width"], env.Dims["height"], 3,))
    layer1 = tf.keras.layers.Conv2D(32, (4, 4), strides=4, activation="relu")(inputs)
    layer2 = tf.keras.layers.Conv2D(64, (4, 4), strides=4, activation="relu")(layer1)
    layer3 = tf.keras.layers.Conv2D(128, (2, 2), strides=2, activation="relu")(layer2)
    
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

def test_game(test_game):
    #first action doesn't matter as it still start of game
    action = np.random.randint(0, no_actions)
    state, game_done, reward, round_val = env.step(action)
    game_done = False
    frame_arr =[]
    
    game_reward = 0
    
    while game_done == False:
        frameIm = 255 * state
        frameIm = frameIm.astype(np.uint8)
        shape = (env.Dims["width"], env.Dims["height"])
        frameIm = frameIm.reshape((*shape, 3))
        frameIm = cv2.cvtColor(frameIm, cv2.COLOR_BGR2RGB)
        frame_arr.append(frameIm)
        
        pred = agent.predict(state.reshape((-1, env.Dims["width"], env.Dims["height"], 3)))[0]
        action = pred.argmax()
        state, game_done, reward, round_val = env.step(action)
        game_reward += reward
        
    sizeFrame = (env.Dims["height"], env.Dims["width"]);
    out = cv2.VideoWriter('logs/DigDug'+ str(int(test_game)) +'_' + str(int(game_reward)) +'.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20, sizeFrame, True)
    
        
    for i in range(len(frame_arr)):
        out.write(frame_arr[i])
    out.release()
    
def random_game():
    game_done = False
    frame_arr =[]
    while game_done == False:
            action = np.random.randint(0, no_actions)
            state, game_done, reward, round_val = env.step(action)
            frameIm = 255 * state
            frameIm = frameIm.astype(np.uint8)
            shape = (env.Dims["width"], env.Dims["height"])
            frameIm = frameIm.reshape((*shape, 3))
            frameIm = cv2.cvtColor(frameIm, cv2.COLOR_BGR2RGB)
            frame_arr.append(frameIm)
            
    sizeFrame = (env.Dims["height"], env.Dims["width"]);
    out = cv2.VideoWriter('logs/RandomDigDug.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20, sizeFrame, True)
    
    
    for i in range(len(frame_arr)):
        out.write(frame_arr[i])
    out.release()
    

roms_path = "roms/"  # Replace this with the path to your ROMs
env = Environment("env1", roms_path)
env.start()
agent = ann_model()
game = 0
game_reward = 0
agent_steps = 0
inital_step = True
writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

random_game()
env.new_game()

while game < 100000:
        
    if inital_step == True:
        action = np.random.randint(0, no_actions)
        state, game_done, reward, round_val = env.step(action)
        previous_state = state;
        agent_steps += 1
        inital_step = False
    
    discount_reward = gamma * np.max(agent.predict(state.reshape((-1, env.Dims["width"], env.Dims["height"], 3))), axis=1, keepdims=True) + reward
    
    loss, grads = loss_with_grads(previous_state, action, discount_reward, agent)
    #write loss every step
    
    with writer.as_default():
     tf.summary.scalar('Loss', loss, agent_steps)
     tf.summary.scalar('Steps_Epsilon', get_epsilon(agent_steps), agent_steps)
     writer.flush()
    
    opt.apply_gradients(zip(grads, agent.trainable_weights))
    
    action = get_action(agent_steps, state)
    previous_state = state
    state, game_done, reward, round_val = env.step(action)
    agent_steps += 1
    game_reward += reward
            
    # action = random.randint(0, no_actions-1)
    # frames, game_done = env.step(action)
    
    # state = state.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work
    # shape = (env.Dims["width"], env.Dims["height"], 3);
    # state = state.reshape((*shape, 1))
    # plt.imshow(np.array(np.squeeze(state)), cmap='gray', vmin = 0, vmax = 1)
    # plt.show()
    if game_done:
        game += 1;
        #reward ,eps, round
        inital_step = True;
        print("game: " + str(game))
        print("reward: " + str(game_reward))
        with writer.as_default():
            tf.summary.scalar('Game_Epsilon', get_epsilon(agent_steps), game)
            tf.summary.scalar('Steps_Reward', game_reward, agent_steps)
            tf.summary.scalar('Game_Reward', game_reward, game)
            tf.summary.scalar('Steps_Round', round_val, agent_steps)
            tf.summary.scalar('Game_Round', round_val, game)
            writer.flush()
        game_reward = 0
        env.new_game()
        if game % 200 == 0:
            test_game((game/200))
            env.new_game()

#every 100 games do 1 test
env.close();