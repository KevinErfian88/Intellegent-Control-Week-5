import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent  # pastikan dqn_agent.py ada di folder yang sama

if __name__ == "__main__":
    # Buat environment dengan visualisasi
    env = gym.make("CartPole-v1", render_mode="human")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Inisialisasi agen
    agent = DQNAgent(state_size, action_size)

    # >>> Jika ingin load model hasil training sebelumnya, aktifkan ini:
    # agent.model.load_weights("dqn_cartpole.h5")

    # Saat testing, pakai epsilon kecil biar hampir selalu exploit
    agent.epsilon = 0.01

    episodes = 25
    for e in range(episodes):
        obs, info = env.reset()
        state = np.atleast_2d(np.array(obs, dtype=np.float32))

        for time in range(500):
            action = agent.act(state)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = np.atleast_2d(np.array(obs, dtype=np.float32))

            if done:
                print(f"Test Episode {e+1}, Score: {time}")
                break

    env.close()
