# ==============================================================================
# SCRIPT UNIFICADO: EXECUÇÃO DAS 4 ERAS DO ALGORITMO (CORRIGIDO PARA INFRA)
# ==============================================================================
import os
import time
import ale_py
import cv2
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from keras import ops
from collections import deque

gym.register_envs(ale_py)
ENV_NAME = "ALE/MsPacman-v5"

# ------------------------------------------------------------------------------
# 🎛️ CONFIGURAÇÃO DE SELEÇÃO DO MODELO
# Escolha qual arquivo exato da sua pasta você deseja assistir jogando:
# 1 = dqn_mspacman.keras          (Era 1: Double DQN Clássico - Frame Único)
# 2 = dqn_mspacman_v2.keras       (Era 2: Reward Shaped + Buffer Stacked 4)
# 3 = dueling_ddqn_mspacman.h5    (Era 3: Transplante Estrutural Puro Visual)
# 4 = dueling_ddqn_hybrid.h5      (Era 4: Híbrido Final Treinado + RAM)
ERA_DO_MODELO = 4
# ------------------------------------------------------------------------------

NOMES_ARQUIVOS = {
    1: "dqn_mspacman.keras",
    2: "dqn_mspacman_v2.keras",
    3: "dueling_ddqn_mspacman.weights.h5",
    4: "dueling_ddqn_mspacman_hybrid.weights.h5"
}

ARQUIVO_MODELO = NOMES_ARQUIVOS[ERA_DO_MODELO]

def get_ram_features(env):
    ale_ram = env.unwrapped.ale.getRAM()
    pacman_x = float(ale_ram[10])
    pacman_y = float(ale_ram[16])
    blinky_x = float(ale_ram[6])
    pinky_x  = float(ale_ram[7])
    inky_x   = float(ale_ram[8])
    sue_x    = float(ale_ram[9])
    return np.array([pacman_x, pacman_y, blinky_x, pinky_x, inky_x, sue_x], dtype=np.float32) / 255.0

def build_network(tipo_era, action_space):
    action_space = int(action_space)
    
    # Modelo 1 espera entrada de frame único (84, 84, 1), os outros esperam (84, 84, 4)
    channels = 1 if tipo_era == 1 else 4
    img_input = layers.Input(shape=(84, 84, channels), name="image_input")
    
    x = layers.Conv2D(32, (8, 8), strides=4, activation="relu", name="conv1")(img_input)
    x = layers.Conv2D(64, (4, 4), strides=2, activation="relu", name="conv2")(x)
    x = layers.Conv2D(64, (3, 3), strides=1, activation="relu", name="conv3")(x)
    conv_output = layers.Flatten()(x)
    
    if tipo_era in [1, 2, 3]:
        # Fluxo de entrada de Imagem Pura (Ajustado para o arquivo weights.h5 sem RAM)
        value_fc = layers.Dense(512, activation="relu")(conv_output)
        value = layers.Dense(1, activation=None)(value_fc)
        
        advantage_fc = layers.Dense(512, activation="relu")(conv_output)
        advantage = layers.Dense(action_space, activation=None)(advantage_fc)
    else:
        # Fluxo Híbrido Final Multi-Input (Fusão Convolução + RAM)
        feat_input = layers.Input(shape=(6,), name="feature_input")
        merged = layers.Concatenate()([conv_output, feat_input])
        
        value_fc = layers.Dense(512, activation="relu")(merged)
        value = layers.Dense(1, activation=None)(value_fc)
        
        advantage_fc = layers.Dense(512, activation="relu")(merged)
        advantage = layers.Dense(action_space, activation=None)(advantage_fc)
        
    mean_advantage = ops.mean(advantage, axis=1, keepdims=True)
    output = value + (advantage - mean_advantage)
    
    if tipo_era in [1, 2, 3]:
        return Model(inputs=img_input, outputs=output)
    else:
        return Model(inputs=[img_input, feat_input], outputs=output)

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))
    return frame / 255.0

# --- INICIALIZAÇÃO ---
print(f"\n[INICIANDO] Executando Opção {ERA_DO_MODELO} -> Arquivo: '{ARQUIVO_MODELO}'")

if not os.path.exists(ARQUIVO_MODELO):
    print(f"[ERRO CRÍTICO] Arquivo '{ARQUIVO_MODELO}' não encontrado!")
    exit(1)

env = gym.make(ENV_NAME, render_mode="human")
action_space_size = int(env.action_space.n)

# Reconstrói a topologia condicional correta antes de acoplar as matrizes
model = build_network(ERA_DO_MODELO, action_space=action_space_size)

if ARQUIVO_MODELO.endswith(".keras"):
    # Carrega diretamente as topologias compiladas do Keras nativo
    model = tf.keras.models.load_model(ARQUIVO_MODELO, compile=False)
else:
    # Carrega as matrizes brutas do formato H5 de forma posicional estável
    model.load_weights(ARQUIVO_MODELO)
print("[OK] Modelo acoplado com sucesso!")

# --- LOOP DE JOGO ---
state, _ = env.reset()
channels_needed = 1 if ERA_DO_MODELO == 1 else 4
img_buffer = deque(maxlen=channels_needed)

processed_img = preprocess_frame(state)
for _ in range(channels_needed):
    img_buffer.append(processed_img)

score_total = 0
running = True

while running:
    if ERA_DO_MODELO == 1:
        # Passa apenas o último frame cinza puro (1, 84, 84, 1)
        state_input = np.expand_dims(img_buffer[-1], axis=(0, -1))
    else:
        # Passa o empilhamento completo de 4 canais (1, 84, 84, 4)
        state_stacked = np.stack(img_buffer, axis=-1)
        state_input = np.expand_dims(state_stacked, axis=0)
    
    if ERA_DO_MODELO in [1, 2, 3]:
        q_values = model(state_input, training=False)
    else:
        ram_features = get_ram_features(env)
        ram_features = np.expand_dims(ram_features, axis=0)
        q_values = model([state_input, ram_features], training=False)
        
    action = int(np.argmax(q_values[0]))
    next_state, reward, terminated, truncated, _ = env.step(action)
    score_total += reward
    img_buffer.append(preprocess_frame(next_state))
    
    if terminated or truncated:
        print(f"\n[FIM] Pontuação do arquivo {ARQUIVO_MODELO}: {score_total} pontos.")
        running = False
        
env.close()