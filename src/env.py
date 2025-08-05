import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PlayerImitationEnv(gym.Env):
    def __init__(self, target_player, ball):
        super(PlayerImitationEnv, self).__init__()
        self.target_player = target_player
        self.ball = ball
        self.max_steps = len(target_player)
        
        # Maksimum hız hesaplama (action space için)
        pos = self.target_player[["position_x", "position_y"]].values
        dx = np.diff(pos[:, 0])
        dy = np.diff(pos[:, 1])
        max_speed = np.max(np.sqrt(dx**2 + dy**2)) if len(dx) > 0 else 10.0
        
        # Action Space: [dx, dy] - oyuncunun bir sonraki adımda hareket edeceği yön
        self.action_space = spaces.Box(
            low=np.array([-max_speed, -max_speed], dtype=np.float32), 
            high=np.array([max_speed, max_speed], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation Space: [agent_x, agent_y, ball_x, ball_y, ball_direction]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        
        # Başlangıç değerleri
        self.current_step = 0
        self.agent_pos = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Simülasyonu baştan başlat
        self.current_step = 0
        
        # Başlangıç pozisyonu
        initial_pos = self.target_player[["position_x", "position_y"]].iloc[0].values
        self.agent_pos = initial_pos.copy()
        
        # NaN kontrolü ve temizleme
        self.agent_pos = np.nan_to_num(self.agent_pos, nan=0.0, posinf=1000.0, neginf=-1000.0)
        
        # İlk gözlem
        ball_pos = self.ball[["position_x", "position_y"]].iloc[0].values
        ball_dir = np.array([self.ball["direction_deg"].iloc[0]])
        
        # NaN kontrolü
        ball_pos = np.nan_to_num(ball_pos, nan=0.0, posinf=1000.0, neginf=-1000.0)
        ball_dir = np.nan_to_num(ball_dir, nan=0.0, posinf=360.0, neginf=0.0)
        
        observation = np.concatenate([self.agent_pos, ball_pos, ball_dir])
        observation = np.nan_to_num(observation, nan=0.0)
        
        info = {}
        return observation.astype(np.float32), info

    def step(self, action):
        # Action'ı temizle
        action = np.nan_to_num(action, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Action'ı uygula (agent'ın yeni pozisyonu)
        self.agent_pos += action
        self.current_step += 1
        
        # Hedef oyuncunun bu adımdaki gerçek pozisyonu
        if self.current_step < len(self.target_player):
            target_pos = self.target_player[["position_x", "position_y"]].iloc[self.current_step].values
            ball_pos = self.ball[["position_x", "position_y"]].iloc[self.current_step].values
            ball_dir = np.array([self.ball["direction_deg"].iloc[self.current_step]])
        else:
            # Veri bittiğinde son değerleri kullan
            target_pos = self.target_player[["position_x", "position_y"]].iloc[-1].values
            ball_pos = self.ball[["position_x", "position_y"]].iloc[-1].values
            ball_dir = np.array([self.ball["direction_deg"].iloc[-1]])
        
        # NaN kontrolü ve temizleme
        target_pos = np.nan_to_num(target_pos, nan=0.0, posinf=1000.0, neginf=-1000.0)
        ball_pos = np.nan_to_num(ball_pos, nan=0.0, posinf=1000.0, neginf=-1000.0)
        ball_dir = np.nan_to_num(ball_dir, nan=0.0, posinf=360.0, neginf=0.0)
        self.agent_pos = np.nan_to_num(self.agent_pos, nan=0.0, posinf=1000.0, neginf=-1000.0)
        
        # Reward hesaplama: Agent pozisyonu ile gerçek oyuncu pozisyonu arasındaki mesafe
        distance_error = np.linalg.norm(self.agent_pos - target_pos)
        distance_error = np.nan_to_num(distance_error, nan=100.0)  # NaN'ı büyük bir hata değeri yap
        
        reward = -distance_error  # Mesafe ne kadar az o kadar iyi
        reward = np.clip(reward, -1000, 1000)  # Reward'ı sınırla
        
        # Episode bitme koşulları
        terminated = self.current_step >= self.max_steps - 1
        truncated = False
        
        # Yeni gözlem
        observation = np.concatenate([self.agent_pos, ball_pos, ball_dir])
        observation = np.nan_to_num(observation, nan=0.0)
        
        info = {
            'distance_error': distance_error,
            'target_pos': target_pos,
            'agent_pos': self.agent_pos.copy()
        }
        
        return observation.astype(np.float32), reward, terminated, truncated, info

    def render(self, mode='human'):
        # Görselleştirme (isteğe bağlı)
        if hasattr(self, '_fig'):
            plt.clf()
        else:
            import matplotlib.pyplot as plt
            self._fig, self._ax = plt.subplots()
        
        # Agent ve hedef pozisyonları çiz
        if self.current_step < len(self.target_player):
            target_pos = self.target_player[["position_x", "position_y"]].iloc[self.current_step].values
            self._ax.scatter(*self.agent_pos, c='red', s=100, label='Agent')
            self._ax.scatter(*target_pos, c='blue', s=100, label='Target')
            self._ax.legend()
            plt.pause(0.01)