import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class Player14ImitationEnv(gym.Env):
    def __init__(self, player14_data, ball_data):
        super(Player14ImitationEnv, self).__init__()
        self.player14_data = player14_data
        self.ball_data = ball_data
        
        # Veri uzunluğunu eşitle (daha kısa olan veriyi baz al)
        min_length = min(len(player14_data), len(ball_data))
        self.player14_data = player14_data.iloc[:min_length].copy()
        self.ball_data = ball_data.iloc[:min_length].copy()
        self.max_steps = min_length - 1
        
        print(f"Player 14 veri uzunluğu: {len(self.player14_data)}")
        print(f"Ball veri uzunluğu: {len(self.ball_data)}")
        print(f"Maksimum episode uzunluğu: {self.max_steps}")
        
        # Player 14'ün maksimum hız hesaplama
        pos = self.player14_data[["position_x", "position_y"]].values
        if len(pos) > 1:
            dx = np.diff(pos[:, 0])
            dy = np.diff(pos[:, 1])
            speeds = np.sqrt(dx**2 + dy**2)
            max_speed = np.max(speeds) if len(speeds) > 0 else 10.0
            avg_speed = np.mean(speeds) if len(speeds) > 0 else 5.0
        else:
            max_speed = 10.0
            avg_speed = 5.0
            
        print(f"Player 14 maksimum hız: {max_speed:.2f}")
        print(f"Player 14 ortalama hız: {avg_speed:.2f}")
        
        # Action Space: [dx, dy] - bir sonraki adımdaki hareket
        self.action_space = spaces.Box(
            low=np.array([-max_speed, -max_speed], dtype=np.float32), 
            high=np.array([max_speed, max_speed], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation Space: [agent_x, agent_y, ball_x, ball_y, ball_direction, prev_dx, prev_dy]
        # Önceki hareket yönü de dahil edildi ki agent daha iyi öğrensin
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        
        self.current_step = 0
        self.agent_pos = None
        self.prev_movement = np.array([0.0, 0.0])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Player 14'ün başlangıç pozisyonu
        initial_pos = self.player14_data[["position_x", "position_y"]].iloc[0].values
        self.agent_pos = initial_pos.copy()
        self.prev_movement = np.array([0.0, 0.0])
        
        # NaN kontrolü
        self.agent_pos = np.nan_to_num(self.agent_pos, nan=0.0)
        
        # İlk gözlem
        ball_pos = self.ball_data[["position_x", "position_y"]].iloc[0].values
        ball_dir = np.array([self.ball_data["direction_deg"].iloc[0]])
        
        # NaN kontrolü
        ball_pos = np.nan_to_num(ball_pos, nan=0.0)
        ball_dir = np.nan_to_num(ball_dir, nan=0.0)
        
        # Observation: [agent_x, agent_y, ball_x, ball_y, ball_direction, prev_dx, prev_dy]
        observation = np.concatenate([
            self.agent_pos, 
            ball_pos, 
            ball_dir,
            self.prev_movement
        ])
        observation = np.nan_to_num(observation, nan=0.0)
        
        info = {'step': self.current_step}
        return observation.astype(np.float32), info

    def step(self, action):
        # Action'ı temizle
        action = np.nan_to_num(action, nan=0.0)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Agent'ı hareket ettir
        self.agent_pos += action
        self.prev_movement = action.copy()
        self.current_step += 1
        
        # Player 14'ün bu adımdaki gerçek pozisyonu
        if self.current_step < len(self.player14_data):
            target_pos = self.player14_data[["position_x", "position_y"]].iloc[self.current_step].values
            ball_pos = self.ball_data[["position_x", "position_y"]].iloc[self.current_step].values
            ball_dir = np.array([self.ball_data["direction_deg"].iloc[self.current_step]])
            
            # Player 14'ün gerçek hareketi
            prev_target_pos = self.player14_data[["position_x", "position_y"]].iloc[self.current_step-1].values
            real_movement = target_pos - prev_target_pos
        else:
            # Son adımda kalmaya devam et
            target_pos = self.player14_data[["position_x", "position_y"]].iloc[-1].values
            ball_pos = self.ball_data[["position_x", "position_y"]].iloc[-1].values
            ball_dir = np.array([self.ball_data["direction_deg"].iloc[-1]])
            real_movement = np.array([0.0, 0.0])
        
        # NaN kontrolü
        target_pos = np.nan_to_num(target_pos, nan=0.0)
        ball_pos = np.nan_to_num(ball_pos, nan=0.0)
        ball_dir = np.nan_to_num(ball_dir, nan=0.0)
        self.agent_pos = np.nan_to_num(self.agent_pos, nan=0.0)
        real_movement = np.nan_to_num(real_movement, nan=0.0)
        
        # REWARD HESAPLAMA
        # 1. Pozisyon hatası (ana reward)
        position_error = np.linalg.norm(self.agent_pos - target_pos)
        position_reward = -position_error
        
        # 2. Hareket yönü benzerliği (bonus reward)
        movement_error = np.linalg.norm(action - real_movement)
        movement_reward = -movement_error * 0.5  # Daha az ağırlıklı
        
        # 3. Toplam reward
        total_reward = position_reward + movement_reward
        total_reward = np.clip(total_reward, -1000, 1000)
        
        # Episode bitme koşulları
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Yeni gözlem
        observation = np.concatenate([
            self.agent_pos, 
            ball_pos, 
            ball_dir,
            self.prev_movement
        ])
        observation = np.nan_to_num(observation, nan=0.0)
        
        info = {
            'step': self.current_step,
            'position_error': position_error,
            'movement_error': movement_error,
            'target_pos': target_pos,
            'agent_pos': self.agent_pos.copy(),
            'real_movement': real_movement,
            'predicted_movement': action
        }
        
        return observation.astype(np.float32), total_reward, terminated, truncated, info