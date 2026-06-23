import os
import time
import ale_py
import cv2
import gymnasium as gym
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, Model
from keras import ops
from langchain_core.runnables import RunnableLambda

# --- CORREÇÃO DO ERRO DO ATARI NO VS CODE ---
gym.register_envs(ale_py)

# --- CONFIGURAÇÕES ---
ENV_NAME = "ALE/MsPacman-v5"

# Descobre a pasta onde este script está salvo para achar o arquivo de pesos no mesmo lugar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRIVE_PATH = os.path.join(BASE_DIR, "dueling_ddqn_mspacman.weights.h5")

print(f"=== Diretorio do Script: {BASE_DIR} ===")
print(f"=== Procurando pesos em: {DRIVE_PATH} ===")

if not os.path.exists(DRIVE_PATH):
    print(f"[ERRO] O arquivo '{DRIVE_PATH}' nao foi encontrado!")
    print("Verifique se o arquivo .weights.h5 esta na mesma pasta que este script Python.")
    exit(1)

# --- 1. FUNÇÃO PARA CONSTRUIR A ARQUITETURA DUELING DQN ---
def build_dueling_dqn(input_shape=(84, 84, 4), action_space=9):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    x = layers.Conv2D(64, 4, strides=2, activation="relu")(x)
    x = layers.Conv2D(64, 3, strides=1, activation="relu")(x)
    x = layers.Flatten()(x)
    
    value_fc = layers.Dense(512, activation="relu")(x)
    value = layers.Dense(1, activation=None)(value_fc)
    
    advantage_fc = layers.Dense(512, activation="relu")(x)
    advantage = layers.Dense(action_space, activation=None)(advantage_fc)
    
    mean_advantage = ops.mean(advantage, axis=1, keepdims=True)
    output = value + (advantage - mean_advantage)
    
    model = Model(inputs=inputs, outputs=output)
    return model

# --- 2. MÓDULO DE PROCESSAMENTO ---
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))
    return frame / 255.0

# --- 3. IMPLEMENTAÇÃO DO FORMATO DE CHAIN ---
class DQNAgentChain:
    def __init__(self, weights_path, action_space):
        print(f"--- Construindo arquitetura Dueling DDQN... ---")
        self.model = build_dueling_dqn(input_shape=(84, 84, 4), action_space=int(action_space))
        
        print(f"--- Carregando pesos do arquivo... ---")
        try:
            self.model.load_weights(weights_path)
            print("[OK] Pesos carregados com sucesso!")
        except Exception as e:
            print(f"[ERRO] Erro critico ao carregar pesos.")
            exit(1)
            
        self.brain_chain = RunnableLambda(self._predict_action)

    def _predict_action(self, state_input):
        q_values = self.model(state_input, training=False)
        return int(np.argmax(q_values[0]))

    def invoke(self, state_input):
        return self.brain_chain.invoke(state_input)

# --- 4. EXECUÇÃO DO FLUXO DO AMBIENTE ---
def run_atari_simulation():
    print("--- Inicializando o ambiente Gymnasium ---")
    try:
        env = gym.make(ENV_NAME, render_mode="human")
    except Exception as e:
        print(f"[AVISO] Erro ao abrir emulador grafico. Tentando em modo cego...")
        env = gym.make(ENV_NAME, render_mode=None)
        
    action_space = int(env.action_space.n) 
    agent_chain = DQNAgentChain(DRIVE_PATH, action_space)
    
    print("--- Partida Iniciada! ---")
    
    for episodio in range(1, 4):
        print(f"\n-> Iniciando Episodio {episodio}")
        state, info = env.reset()

        state_deque = deque(maxlen=4)
        first_frame = preprocess_frame(state)
        for _ in range(4):
            state_deque.append(first_frame)

        done = False
        total_reward = 0
        passos = 0

        while not done:
            state_stacked = np.stack(state_deque, axis=-1)
            state_input = np.expand_dims(state_stacked, axis=0)

            if passos == 0:
                action = env.action_space.sample()
            else:
                action = agent_chain.invoke(state_input)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            next_frame = preprocess_frame(next_state)
            state_deque.append(next_frame)

            total_reward += reward
            passos += 1
            
            if passos % 50 == 0:
                print(f"   -> Rodando... Passo: {passos} | Pontuacao Atual: {total_reward}")
                
            time.sleep(0.02)

        print(f"[-] Fim do Episodio {episodio}! Passos: {passos} | Pontuacao: {total_reward}")
        time.sleep(1.0)
        
    env.close()

if __name__ == "__main__":
    run_atari_simulation()