import numpy as np
import cv2
import tensorflow as tf
from Environment import Environment

no_actions = 5

min_epsilon = 0.1
max_epsilon = 0.99 
decay = 0.0015 
gamma = 0.99
alpha = 0.0001
buff_state = []
buff_action = []
buff_reward = []
BATCH_SIZE = 32
BUFF_MAX = 5500
TENSORBOARD_DIR = 'logs/tensorboard/'
RL_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)

opt = tf.optimizers.Adam(alpha)

def get_epsilon(game):
    eps = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * game)
    return eps

def loss_with_grads(prev_state, action, state, reward, model):
    with tf.GradientTape(persistent=True) as tape:
        mask = tf.one_hot(action, no_actions, dtype=tf.float32)
        
        target = (gamma * tf.reduce_max(model(state), axis=1, keepdims=True) + reward) * mask
        
        pred = model(prev_state) * mask
        
        loss = tf.reduce_mean(tf.reduce_sum(RL_loss(target,pred), axis=-1))
    
    grads = tape.gradient(loss,model.trainable_weights)
    
    return loss, grads

def ann_model():
    inputs = tf.keras.layers.Input(shape=(env.Dims["width"], env.Dims["height"], 3,))
    net = tf.keras.layers.Conv2D(32, (4, 4), strides=(4,4), activation="elu", padding='same')(inputs)
    net = tf.keras.layers.Conv2D(64, (2, 2), activation="elu", padding='same')(net)
    net = tf.keras.layers.MaxPool2D()(net)
    net = tf.keras.layers.Conv2D(128, (2, 2), activation="elu", padding='same')(net)
    net = tf.keras.layers.MaxPool2D()(net)
    net = tf.keras.layers.Conv2D(256, (2, 2), activation="elu", padding='same')(net)
    net = tf.keras.layers.MaxPool2D()(net)
    net = tf.keras.layers.Conv2D(512, (2, 2), activation="elu", padding='same')(net)
    net = tf.keras.layers.MaxPool2D()(net)
    
    val_stream, adv_stream  = tf.split(net, 2, 3)

    val_stream = tf.keras.layers.Flatten()(val_stream)
    val_stream = tf.keras.layers.Dense(1, activation="linear")(val_stream)
    
    adv_stream = tf.keras.layers.Flatten()(adv_stream)
    adv_stream = tf.keras.layers.Dense(no_actions, activation="linear")(adv_stream)
    
    q_vals = val_stream + (adv_stream - tf.reduce_max(adv_stream, axis=1, keepdims=True))
    
    # Build model
    model = tf.keras.Model(inputs, q_vals)
    
    model.summary()
    
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
        #print(pred,action)
        
        state, game_done, reward, round_val = env.step(action)
        game_reward += reward
        
    sizeFrame = (env.Dims["height"], env.Dims["width"]);
    out = cv2.VideoWriter('logs/DigDug'+ str(int(test_game)) +'_' + str(int(game_reward)) +'.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20, sizeFrame, True)
    
    with writer.as_default():
     tf.summary.scalar('Test_Reward', game_reward, test_game)
     writer.flush()
    
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
    

roms_path = "roms/"
env = Environment("env1", roms_path)
env.start()
agent = ann_model()
game = 1
game_reward = 0
agent_steps = 0
inital_step = True
writer = tf.summary.create_file_writer(TENSORBOARD_DIR)
learning_steps = 0

random_game()
env.new_game()

while game < 100000:
    if inital_step == True:
        action = np.random.randint(0, no_actions)
        state, game_done, reward, round_val = env.step(action)
        agent_steps += 1
        inital_step = False
    
    buff_state.append(state.reshape((env.Dims["width"], env.Dims["height"], 3)))
    buff_action.append(action)
    buff_reward.append(reward)
    
    if len(buff_state) > BUFF_MAX:
        buff_state = buff_state[1:]
        buff_action = buff_action[1:]
        buff_reward = buff_reward[1:]
    
    if learning_steps == BATCH_SIZE:
        n = np.minimum(len(buff_state)-1,BATCH_SIZE)+1
        idx = np.random.choice(list(range(len(buff_state))),n,False)
        
        batch_state = np.stack([buff_state[i] for i in idx]).reshape((n,env.Dims["width"], env.Dims["height"], 3))
        batch_reward = np.array([buff_reward[i] for i in idx]).reshape((n,1))
        batch_action = [buff_action[i] for i in idx]
        batch_prev_state = np.stack([buff_state[i-1] for i in idx]).reshape((n,env.Dims["width"], env.Dims["height"], 3))
        
        loss, grads = loss_with_grads(batch_prev_state, batch_action, batch_state, batch_reward, agent)
        opt.apply_gradients(zip(grads, agent.trainable_weights))
        
        with writer.as_default():
         tf.summary.scalar('Loss', loss, agent_steps)
         writer.flush()
        
        learning_steps = 0
    else:
        learning_steps += 1
    
    if np.random.rand(1) < get_epsilon(game):
        action = np.random.randint(0, no_actions)
    else:
        pred = agent.predict(state.reshape((-1, env.Dims["width"], env.Dims["height"], 3)))[0]
        action = pred.argmax()
    
    state, game_done, reward, round_val = env.step(action)
    agent_steps += 1
    game_reward += reward
    
    if game_done:
        inital_step = True;
        print("game: " + str(game))
        print("reward: " + str(game_reward))
        with writer.as_default():
            tf.summary.scalar('Game_Epsilon', get_epsilon(game), game)
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
        game += 1;

#every 200 games do 1 test
env.close();