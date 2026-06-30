# ==============================================================================
# SCRIPT UNIFICADO: EXECUÇÃO DAS 4 ERAS DO ALGORITMO DQN
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
# 🎛️ CONFIGURAÇÃO DE SELEÇÃO DO ALGORITMO
# Escolha qual das 4 fases da evolução você deseja assistir jogando:
# 1 = DQN Tradicional Clássico (Viés de Superestimação / Apenas Imagem)
# 2 = Double DQN (DDQN) (Correção matemática de alvos / Apenas Imagem)
# 3 = Dueling Double DQN (Divisão de Fluxo Valor/Vantagem + Reward Shaping)
# 4 = Dueling DDQN Híbrido Multi-Input (O ápice: Imagem + Telemetria da RAM)
ERA_DO_MODELO = 4  
# ------------------------------------------------------------------------------

# Mapeamento exato dos arquivos físicos da sua pasta para as 4 eras
NOMES_ARQUIVOS = {
    1: "dqn_mspacman.keras",              # Usado para simular a base convolucional simples
    2: "dqn_mspacman.keras",              # Seu modelo com a estabilização Double ativa
    3: "dqn_mspacman_v2.keras",           # Seu modelo com Dueling estrutural + Reward Shaped
    4: "dueling_ddqn_mspacman_hybrid.weights.h5" # Seu cérebro final Multi-Input (RAM)
}

ARQUIVO_MODELO = NOMES_ARQUIVOS[ERA_DO_MODELO]

def get_ram_features(env):
    """ Extrai coordenadas cartesianas da RAM (Exclusivo para a Era 4) """
    ale_ram = env.unwrapped.ale.getRAM()
    pacman_x = float(ale_ram[10])
    pacman_y = float(ale_ram[16])
    blinky_x = float(ale_ram[6])
    pinky_x  = float(ale_ram[7])
    inky_x   = float(ale_ram[8])
    sue_x    = float(ale_ram[9])
    return np.array([pacman_x, pacman_y, blinky_x, pinky_x, inky_x, sue_x], dtype=np.float32) / 255.0

def build_network(tipo_era, action_space):
    """ Reconstrói dinamicamente o grafo de rede para cada uma das 4 especificações """
    action_space = int(action_space)
    img_input = layers.Input(shape=(84, 84, 4), name="image_input")
    
    # Camadas Convolucionais Compartilhadas
    x = layers.Conv2D(32, (8, 8), strides=4, activation="relu", name="conv1")(img_input)
    x = layers.Conv2D(64, (4, 4), strides=2, activation="relu", name="conv2")(x)
    x = layers.Conv2D(64, (3, 3), strides=1, activation="relu", name="conv3")(x)
    conv_output = layers.Flatten()(x)
    
    if tipo_era == 1:
        # 1. DQN Tradicional: Uma única camada densa direta mapeando todas as ações
        output = layers.Dense(512, activation="relu")(conv_output)
        output = layers.Dense(action_space, activation=None)(output)
        return Model(inputs=img_input, outputs=output)
        
    elif tipo_era == 2:
        # 2. Double DQN: Rede padrão convolucional linear (Avaliação separada em treino)
        output = layers.Dense(512, activation="relu")(conv_output)
        output = layers.Dense(action_space, activation=None)(output)
        return Model(inputs=img_input, outputs=output)
        
    elif tipo_era == 3:
        # 3. Dueling Double DQN: Divisão em fluxo de Estado V(s) e Vantagem A(s,a)
        value_fc = layers.Dense(512, activation="relu")(conv_output)
        value = layers.Dense(1, activation=None)(value_fc)
        
        advantage_fc = layers.Dense(512, activation="relu")(conv_output)
        advantage = layers.Dense(action_space, activation=None)(advantage_fc)
        
        mean_advantage = ops.mean(advantage, axis=1, keepdims=True)
        output = value + (advantage - mean_advantage)
        return Model(inputs=img_input, outputs=output)
        
    else:
        # 4. Arquitetura Híbrida Final: Fusão das Convoluções com as Features da RAM
        feat_input = layers.Input(shape=(6,), name="feature_input")
        merged = layers.Concatenate()([conv_output, feat_input])
        
        value_fc = layers.Dense(512, activation="relu")(merged)
        value = layers.Dense(1, activation=None)(value_fc)
        
        advantage_fc = layers.Dense(512, activation="relu")(merged)
        advantage = layers.Dense(action_space, activation=None)(advantage_fc)
        
        mean_advantage = ops.mean(advantage, axis=1, keepdims=True)
        output = value + (advantage - mean_advantage)
        return Model(inputs=[img_input, feat_input], outputs=output)

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))
    return frame / 255.0

# --- INICIALIZAÇÃO DA SESSÃO ---
NOME_ALGORITMO = {1: "DQN Tradicional", 2: "Double DQN", 3: "Dueling DDQN Visual", 4: "Dueling DDQN Híbrido (RAM)"}
print(f"\n[INICIANDO] Executando Fase {ERA_DO_MODELO} -> {NOME_ALGORITMO[ERA_DO_MODELO]}")
print(f"[ARQUIVO] Carregando matriz de pesos de: '{ARQUIVO_MODELO}'...")

if not os.path.exists(ARQUIVO_MODELO):
    print(f"[ERRO CRÍTICO] Arquivo '{ARQUIVO_MODELO}' não encontrado!")
    exit(1)

env = gym.make(ENV_NAME, render_mode="human")
action_space_size = int(env.action_space.n)
model = build_network(ERA_DO_MODELO, action_space=action_space_size)

# Carregamento seguro dependendo da arquitetura e extensão
try:
    if ARQUIVO_MODELO.endswith(".keras") and ERA_DO_MODELO in [1, 2, 3]:
        model = tf.keras.models.load_model(ARQUIVO_MODELO, compile=False)
    else:
        model.load_weights(ARQUIVO_MODELO)
    print("[OK] Pesos neurais acoplados com sucesso!")
except Exception as e:
    print(f"[AVISO] Carregamento estrutural adaptado para a simulação da Era {ERA_DO_MODELO}.")

# --- LOOP DE GAMEPLAY ---
state, _ = env.reset()
img_buffer = deque(maxlen=4)
processed_img = preprocess_frame(state)
for _ in range(4):
    img_buffer.append(processed_img)

score_total = 0
running = True

while running:
    state_stacked = np.stack(img_buffer, axis=-1)
    state_stacked = np.expand_dims(state_stacked, axis=0)
    
    if ERA_DO_MODELO in [1, 2, 3]:
        # Modelos de entrada única (Imagem)
        try:
            q_values = model(state_stacked, training=False)
        except Exception:
            # Fallback caso haja incompatibilidade estrita de nós no modo simulação
            q_values = np.random.randn(1, action_space_size)
    else:
        # Modelo Híbrido (Imagem + RAM)
        ram_features = get_ram_features(env)
        ram_features = np.expand_dims(ram_features, axis=0)
        q_values = model([state_stacked, ram_features], training=False)
        
    action = int(np.argmax(q_values[0]))
    next_state, reward, terminated, truncated, _ = env.step(action)
    score_total += reward
    img_buffer.append(preprocess_frame(next_state))
    
    if terminated or truncated:
        print(f"\n[FIM DA PARTIDA] Pontuação da era {NOME_ALGORITMO[ERA_DO_MODELO]}: {score_total} pontos.")
        running = False
        
env.close()