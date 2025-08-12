import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
import torch.nn as nn 

class BalancedPlayer14Env(gym.Env):
    def __init__(self, player14_data, ball_data):
        super(BalancedPlayer14Env, self).__init__()
        self.player14_data = player14_data
        self.ball_data = ball_data
        
        # Veri uzunluğunu eşitle
        min_length = min(len(player14_data), len(ball_data))
        self.player14_data = player14_data.iloc[:min_length].copy()
        self.ball_data = ball_data.iloc[:min_length].copy()
        self.max_steps = min_length - 1
        
        print(f"Player 14 veri uzunluğu: {len(self.player14_data)}")
        print(f"Ball veri uzunluğu: {len(self.ball_data)}")
        
        # 🎯 BALANCED ACTION SPACE - Gerçek hareketlere dayalı smooth action space
        pos = self.player14_data[["position_x", "position_y"]].values
        if len(pos) > 1:
            dx = np.diff(pos[:, 0])
            dy = np.diff(pos[:, 1])
            speeds = np.sqrt(dx**2 + dy**2)
            
            # Gerçek hareket istatistikleri
            mean_speed = np.mean(speeds[speeds > 0.01])  # Anlamlı hareketler
            std_speed = np.std(speeds[speeds > 0.01])
            max_speed = np.percentile(speeds, 95)  # %95 percentile (aşırı değerleri alma)
            
            print(f"Player 14 ortalama hız: {mean_speed:.3f}")
            print(f"Player 14 hız std: {std_speed:.3f}")
            print(f"Player 14 95th percentile hız: {max_speed:.3f}")
            
            # Action space - gerçek hızın 1.75 katı (daha dengeli)
            self.speed_limit = mean_speed * 1.75
            self.min_movement = mean_speed * 0.05  # Çok düşük minimum hareket
            self.target_speed = mean_speed  # Hedef hız
            
        else:
            self.speed_limit = 8.0
            self.min_movement = 0.3
            self.target_speed = 3.0
        
        print(f"Action space hız limiti: {self.speed_limit:.3f}")
        print(f"Minimum hareket: {self.min_movement:.3f}")
        print(f"Hedef hız: {self.target_speed:.3f}")
        
        # Smooth action space
        self.action_space = spaces.Box(
            low=np.array([-self.speed_limit, -self.speed_limit], dtype=np.float32), 
            high=np.array([self.speed_limit, self.speed_limit], dtype=np.float32),
            dtype=np.float32
        )
        
        # 🎯 BALANCED OBSERVATION SPACE 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32
        )
        
        self.current_step = 0
        self.agent_pos = None
        self.agent_velocity = np.array([0.0, 0.0])
        self.prev_movements = [np.array([0.0, 0.0]) for _ in range(3)]
        self.total_distance_traveled = 0.0
        self.stationary_count = 0  # Durağanlık sayacı
        
        # Normalizasyon için scale faktörleri
        pos_range = np.ptp(pos, axis=0)
        self.pos_scale = 1.0 / (np.max(pos_range) + 1e-8) if len(pos) > 0 else 1e-3
        self.max_distance = np.linalg.norm(pos_range) if len(pos) > 0 else 1000.0
        
        print(f"Pozisyon scale faktörü: {self.pos_scale:.6f}")
        print(f"Maksimum mesafe: {self.max_distance:.2f}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.total_distance_traveled = 0.0
        self.stationary_count = 0
        
        # Player 14'ün başlangıç pozisyonu
        initial_pos = self.player14_data[["position_x", "position_y"]].iloc[0].values
        self.agent_pos = np.nan_to_num(initial_pos.copy(), nan=0.0)
        self.agent_velocity = np.array([0.0, 0.0])
        self.prev_movements = [np.array([0.0, 0.0]) for _ in range(3)]
        
        observation = self._get_observation()
        info = {'step': self.current_step}
        return observation, info
    
    def _get_observation(self):
        """🎯 BALANCED OBSERVATION - Smooth decision making için"""
        # Mevcut ball pozisyonu ve hızı
        if self.current_step < len(self.ball_data):
            ball_pos = self.ball_data[["position_x", "position_y"]].iloc[self.current_step].values
            ball_dir = self.ball_data["direction_deg"].iloc[self.current_step]
            
            # Ball velocity (bir sonraki adımdan hesapla)
            if self.current_step + 1 < len(self.ball_data):
                next_ball_pos = self.ball_data[["position_x", "position_y"]].iloc[self.current_step + 1].values
                ball_velocity = next_ball_pos - ball_pos
            else:
                ball_velocity = np.array([0.0, 0.0])
        else:
            ball_pos = self.ball_data[["position_x", "position_y"]].iloc[-1].values
            ball_dir = self.ball_data["direction_deg"].iloc[-1]
            ball_velocity = np.array([0.0, 0.0])
        
        # NaN kontrolü
        ball_pos = np.nan_to_num(ball_pos, nan=0.0)
        ball_dir = np.nan_to_num(ball_dir, nan=0.0)
        ball_velocity = np.nan_to_num(ball_velocity, nan=0.0)
        
        # Göreceli ball pozisyonu
        relative_ball = ball_pos - self.agent_pos
        
        # Mevcut ve gelecekteki target pozisyonları
        current_target = self.player14_data[["position_x", "position_y"]].iloc[self.current_step].values
        if self.current_step + 1 < len(self.player14_data):
            next_target = self.player14_data[["position_x", "position_y"]].iloc[self.current_step + 1].values
            target_velocity = next_target - current_target
        else:
            next_target = current_target.copy()
            target_velocity = np.array([0.0, 0.0])
        
        current_target = np.nan_to_num(current_target, nan=0.0)
        next_target = np.nan_to_num(next_target, nan=0.0)
        target_velocity = np.nan_to_num(target_velocity, nan=0.0)
        
        # Agent'ın hedefle ilişkisi
        distance_to_target = np.linalg.norm(current_target - self.agent_pos)
        direction_to_target = current_target - self.agent_pos
        if distance_to_target > 1e-6:
            direction_to_target = direction_to_target / distance_to_target
        else:
            direction_to_target = np.array([0.0, 0.0])
        
        # Hedefe ulaşmak için gereken smooth hız
        required_velocity = current_target - self.agent_pos
        # Hızı sınırla (smooth movement için)
        required_speed = np.linalg.norm(required_velocity)
        if required_speed > self.target_speed:
            required_velocity = required_velocity / required_speed * self.target_speed
        
        # İlerleme ve zaman bilgileri
        progress_ratio = self.current_step / max(self.max_steps, 1)
        steps_remaining_ratio = (self.max_steps - self.current_step) / max(self.max_steps, 1)
        
        # Gelecek 3 hedef pozisyon (smooth trajectory planning için)
        future_targets_x = []
        future_targets_y = []
        for i in range(3):
            future_idx = min(self.current_step + i + 1, len(self.player14_data) - 1)
            future_target = self.player14_data[["position_x", "position_y"]].iloc[future_idx].values
            future_target = np.nan_to_num(future_target, nan=0.0)
            future_targets_x.append(future_target[0])
            future_targets_y.append(future_target[1])
        
        # Observation vektörü oluştur
        observation_parts = [
            self.agent_pos * self.pos_scale,                    # Agent pozisyonu (2)
            self.agent_velocity * self.pos_scale,               # Agent hızı (2)
            ball_pos * self.pos_scale,                          # Ball pozisyonu (2)
            [ball_dir / 360.0],                                 # Ball yönü (1)
            ball_velocity * self.pos_scale,                     # Ball hızı (2)
            relative_ball * self.pos_scale,                     # Göreceli ball pozisyonu (2)
            current_target * self.pos_scale,                    # Mevcut hedef (2)
            target_velocity * self.pos_scale,                   # Hedef hızı (2)
            [distance_to_target * self.pos_scale],              # Hedefe mesafe (1)
            direction_to_target,                                # Hedef yönü (2)
            required_velocity * self.pos_scale,                 # Gereken smooth hız (2)
            [progress_ratio],                                   # İlerleme oranı (1)
            [steps_remaining_ratio],                            # Kalan zaman oranı (1)
            future_targets_x,                                   # Gelecek hedefler X (3)
            future_targets_y,                                   # Gelecek hedefler Y (3)
            [self.stationary_count / 20.0]                     # Durağanlık faktörü (1)
        ]
        
        # Tüm parçaları birleştir
        observation = np.concatenate(observation_parts)
        
        # Boyut kontrolü
        if len(observation) != 29:
            if len(observation) < 29:
                observation = np.pad(observation, (0, 29 - len(observation)))
            else:
                observation = observation[:29]
        
        return np.nan_to_num(observation, nan=0.0).astype(np.float32)

    def step(self, action):
        # Action'ı temizle ve sınırla
        action = np.nan_to_num(action, nan=0.0)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 🎯 BALANCED SPEED CONTROL - Hedef hızı teşvik et
        action_speed = np.linalg.norm(action)
        
        # Hedef duruyorsa agent da durabilir (küçük adımlamalar)
        current_target = self.player14_data[["position_x", "position_y"]].iloc[self.current_step].values
        if self.current_step > 0:
            prev_target = self.player14_data[["position_x", "position_y"]].iloc[self.current_step-1].values
            real_movement = current_target - prev_target
            real_speed = np.linalg.norm(real_movement)
        else:
            real_movement = np.array([0.0, 0.0])
            real_speed = 0.0
        
        # NaN kontrolü
        current_target = np.nan_to_num(current_target, nan=0.0)
        real_movement = np.nan_to_num(real_movement, nan=0.0)
        
        # Hedef duruyorsa (çok küçük hareket) agent da küçük adımlar atabilir
        if real_speed < self.min_movement:
            # Hedef duruyor, agent de minimal hareket yapabilir
            if action_speed < self.min_movement * 2:
                # İzin verilen minimal hareket
                self.stationary_count += 1
            else:
                # Hedef duruyorken çok hızlı hareket cezası
                self.stationary_count += 2
        else:
            # Hedef hareket ediyor, agent de hareket etmeli
            if action_speed < self.min_movement:
                # Hedef hareket ederken durmak cezası
                if action_speed > 1e-6:
                    # Küçük hareketi hedef yönüne çevir
                    direction = current_target - self.agent_pos
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm > 1e-6:
                        action = direction / direction_norm * self.min_movement
                    else:
                        action = np.array([self.min_movement, 0.0])
                else:
                    # Tamamen sıfırsa, rastgele minimal hareket
                    angle = np.random.uniform(0, 2*np.pi)
                    action = np.array([np.cos(angle), np.sin(angle)]) * self.min_movement
                self.stationary_count += 1
            else:
                self.stationary_count = max(0, self.stationary_count - 1)
        
        # Agent'ı hareket ettir
        old_pos = self.agent_pos.copy()
        self.agent_pos += action
        self.agent_velocity = action.copy()
        
        # Hareket geçmişini güncelle
        self.prev_movements = self.prev_movements[1:] + [action.copy()]
        self.total_distance_traveled += np.linalg.norm(action)
        self.current_step += 1
        
        # 🎯 BALANCED REWARD SYSTEM - Smooth ve gerçekçi davranış teşviki
        
        # 1. Pozisyon doğruluğu (ANA HEDEF) - Ağırlık: 40%
        position_error = np.linalg.norm(self.agent_pos - current_target)
        max_possible_error = self.max_distance
        normalized_error = position_error / max_possible_error
        position_reward = -normalized_error * 30.0  # Daha dengeli ceza
        
        # 2. İlerleme bonusu - Hedefe yaklaşma
        if hasattr(self, 'prev_position_error'):
            progress = self.prev_position_error - position_error
            if progress > 0:
                progress_reward = progress * 50.0  # Yaklaşma bonusu
            else:
                progress_reward = progress * 75.0  # Uzaklaşma cezası
        else:
            progress_reward = 0.0
        self.prev_position_error = position_error
        
        # 3. Hareket yönü uyumu - Gerçek harekete benzerlik (Ağırlık: 25%)
        if np.linalg.norm(real_movement) > 1e-6 and np.linalg.norm(action) > 1e-6:
            real_normalized = real_movement / np.linalg.norm(real_movement)
            action_normalized = action / np.linalg.norm(action)
            movement_similarity = np.dot(real_normalized, action_normalized)
            movement_reward = movement_similarity * 20.0  # Yön uyumu bonusu
        else:
            # Her ikisi de duruyorsa bu iyidir
            if np.linalg.norm(real_movement) < 1e-6 and np.linalg.norm(action) < self.min_movement * 2:
                movement_reward = 10.0  # Hedefle birlikte durma bonusu
            else:
                movement_reward = -5.0  # Uyumsuzluk cezası
            movement_similarity = 0.0
        
        # 4. HIZ UYUMU - EN ÖNEMLİ YENİ KISIM (Ağırlık: 35%)
        agent_speed = np.linalg.norm(action)
        
        if real_speed > self.min_movement:  # Hedef hareket ediyor
            # Hedef hızına yakınlık teşviki
            speed_ratio = agent_speed / max(real_speed, 1e-6)
            if 0.8 <= speed_ratio <= 1.2:  # İdeal hız aralığı
                speed_reward = 25.0
            elif 0.6 <= speed_ratio <= 1.5:  # Kabul edilebilir
                speed_reward = 15.0 - abs(speed_ratio - 1.0) * 10.0
            elif speed_ratio > 2.0:  # Çok hızlı - BÜYÜK CEZA
                speed_reward = -30.0 - (speed_ratio - 2.0) * 20.0
            else:  # Çok yavaş veya çok hızlı
                speed_reward = -abs(speed_ratio - 1.0) * 15.0
        else:  # Hedef duruyor
            if agent_speed < self.target_speed * 0.3:  # Agent de duruyor/yavaş
                speed_reward = 15.0  # Hedefle birlikte durma bonusu
            elif agent_speed < self.target_speed:  # Küçük adımlar
                speed_reward = 10.0 - agent_speed * 5.0
            else:  # Hedef dururken çok hızlı hareket
                speed_reward = -25.0 - agent_speed * 10.0
        
        # 5. Smooth movement teşviki (jerk cezası)
        if len(self.prev_movements) >= 2:
            # Son iki hareket arasındaki smooth geçiş
            prev_action = self.prev_movements[-1]
            curr_action = action
            
            # Hız değişimi (jerk)
            speed_change = abs(np.linalg.norm(curr_action) - np.linalg.norm(prev_action))
            if speed_change < self.target_speed * 0.5:  # Smooth geçiş
                smooth_reward = 8.0 - speed_change * 5.0
            else:  # Ani hız değişimi
                smooth_reward = -speed_change * 8.0
            
            # Yön değişimi
            if np.linalg.norm(prev_action) > 1e-6 and np.linalg.norm(curr_action) > 1e-6:
                prev_normalized = prev_action / np.linalg.norm(prev_action)
                curr_normalized = curr_action / np.linalg.norm(curr_action)
                direction_similarity = np.dot(prev_normalized, curr_normalized)
                if direction_similarity > 0.7:  # Smooth yön geçişi
                    smooth_reward += 5.0
                elif direction_similarity < 0:  # Zıt yön (çok kötü)
                    smooth_reward -= 15.0
        else:
            smooth_reward = 0.0
        
        # 6. Mesafe verimliliği - Gerçek vs agent yol karşılaştırması
        if hasattr(self, 'prev_agent_pos') and hasattr(self, 'prev_real_pos'):
            real_distance = np.linalg.norm(current_target - self.prev_real_pos)
            agent_distance = np.linalg.norm(self.agent_pos - self.prev_agent_pos)
            
            if real_distance > 1e-6:
                distance_ratio = agent_distance / real_distance
                if 0.9 <= distance_ratio <= 1.3:  # Optimal mesafe
                    efficiency_reward = 10.0
                elif distance_ratio <= 2.5:  # Kabul edilebilir
                    efficiency_reward = 5.0 - (distance_ratio - 1.0) * 8.0
                else:  # Çok inefficient
                    efficiency_reward = -20.0 - (distance_ratio - 2.5) * 15.0
            else:
                # Hedef duruyor
                if agent_distance < self.target_speed * 0.5:
                    efficiency_reward = 8.0
                else:
                    efficiency_reward = -agent_distance * 10.0
        else:
            efficiency_reward = 0.0
        
        # 7. Yakınlık bonusu (hedefe yakınsa)
        if position_error < 1.0:
            proximity_bonus = (1.0 - position_error) * 20.0
        elif position_error < 3.0:
            proximity_bonus = (3.0 - position_error) * 5.0
        else:
            proximity_bonus = -5.0
        
        # 8. Durağanlık kontrolü
        if self.stationary_count > 10:  # Uzun süre problem varsa
            stagnation_penalty = -self.stationary_count * 2.0
        else:
            stagnation_penalty = 0.0
        
        # 9. Başarı bonusu
        success_bonus = 0.0
        if position_error < 0.5:
            success_bonus = 100.0
        elif position_error < 1.0:
            success_bonus = 50.0
        elif position_error < 2.0:
            success_bonus = 20.0
        
        # TOPLAM REWARD - Daha dengeli
        total_reward = (
            position_reward +      # Ana hedef
            progress_reward +      # İlerleme
            movement_reward +      # Yön uyumu  
            speed_reward +         # HIZ UYUMU (EN ÖNEMLİ)
            smooth_reward +        # Smooth movement
            efficiency_reward +    # Mesafe verimliliği
            proximity_bonus +      # Yakınlık
            stagnation_penalty +   # Durağanlık cezası
            success_bonus          # Başarı bonusu
        )
        
        # Reward'ı sınırla
        total_reward = np.clip(total_reward, -500, 500)
        
        # Episode bitme koşulları
        success_threshold = 1.0
        terminated = False
        
        if position_error < success_threshold:
            total_reward += 200.0
            terminated = True
            print(f"🎯 BAŞARI! Adım {self.current_step}'te hedefe ulaşıldı (hata: {position_error:.3f})")
        elif self.current_step >= self.max_steps:
            terminated = True
            final_bonus = max(0, 20 - position_error)
            total_reward += final_bonus
        elif position_error > self.max_distance * 1.5:  # Çok uzaklaştıysa
            terminated = True
            total_reward -= 200.0
            print(f"⚠️ Episode erken bitti - çok uzaklaştı (hata: {position_error:.3f})")
        
        truncated = False
        
        # Observation güncelle
        observation = self._get_observation()
        
        # Bir sonraki adım için kaydet
        self.prev_agent_pos = self.agent_pos.copy()
        self.prev_real_pos = current_target.copy()
        
        # Detaylı info
        info = {
            'step': self.current_step,
            'position_error': position_error,
            'movement_similarity': movement_similarity,
            'target_pos': current_target,
            'agent_pos': self.agent_pos.copy(),
            'real_movement': real_movement,
            'predicted_movement': action,
            'real_speed': real_speed,
            'agent_speed': agent_speed,
            'speed_ratio': agent_speed / max(real_speed, 1e-6),
            'position_reward': position_reward,
            'progress_reward': progress_reward,
            'movement_reward': movement_reward,
            'speed_reward': speed_reward,
            'smooth_reward': smooth_reward,
            'efficiency_reward': efficiency_reward,
            'proximity_bonus': proximity_bonus,
            'success_bonus': success_bonus,
            'total_distance_traveled': self.total_distance_traveled,
            'stationary_count': self.stationary_count,
            'success': position_error < success_threshold
        }
        
        return observation, total_reward, terminated, truncated, info

def load_and_clean_data():
    """Player 14 ve ball verisini yükle ve temizle"""
    print("=== VERİ YÜKLEME ===")
    
    try:
        ball = pd.read_csv("ball_clean.csv")
        player14 = pd.read_csv("tracker_14_clean.csv")
    except:
        try:
            ball = pd.read_csv("data/ball_clean.csv")
            player14 = pd.read_csv("data/tracker_14_clean.csv")
        except:
            print("❌ CSV dosyaları bulunamadı! Lütfen dosya yollarını kontrol edin.")
            return None, None
    
    print(f"Ball veri boyutu: {ball.shape}")
    print(f"Player 14 veri boyutu: {player14.shape}")
    
    # Veri temizleme
    def clean_dataframe(df):
        df = df.ffill().bfill().fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        return df
    
    ball = clean_dataframe(ball)
    player14 = clean_dataframe(player14)
    
    print("✅ Veriler temizlendi")
    return ball, player14

def make_env(player14, ball):
    """Environment wrapper"""
    def _init():
        return BalancedPlayer14Env(player14, ball)
    return _init

def train_balanced_agent(timesteps=200000):
    """🎯 BALANCED Player 14 Agent Eğitimi"""
    print(f"=== BALANCED PLAYER 14 AGENT EĞİTİMİ ({timesteps} steps) ===")
    
    # Veri yükleme
    ball, player14 = load_and_clean_data()
    if ball is None or player14 is None:
        return None, None, None
    
    # Environment oluşturma
    env = DummyVecEnv([make_env(player14, ball)])

    # Balanced network architecture
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256, 128, 64],  # Daha küçük ama etkili
                      vf=[256, 256, 128, 64]),
        activation_fn=nn.Tanh  # Smooth activation
    )
    
    # Balanced öğrenme parametreleri
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,      # Dengeli learning rate
        n_steps=1024,            # Daha büyük batch
        batch_size=64,           # Dengeli batch size
        n_epochs=10,             # Moderate epoch
        gamma=0.99,              # Uzun vadeli odaklanma
        gae_lambda=0.95,         # Standard GAE
        clip_range=0.2,          # Conservative clip range
        ent_coef=0.01,           # Düşük exploration (smooth için)
        vf_coef=0.5,             # Balanced value function
        max_grad_norm=0.5,       # Gradient clipping
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tensorboard_logs_balanced/"
    )
    
    # Evaluation callback
    eval_env = DummyVecEnv([make_env(player14, ball)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models_balanced/",
        log_path="./logs_balanced/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Eğitim başlat
    print(f"🎯 Balanced eğitim başlıyor ({timesteps} timesteps)...")
    model.learn(
        total_timesteps=timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Model kaydetme
    model.save("balanced_player14_model")
    print("✅ Balanced model kaydedildi: balanced_player14_model.zip")
    
    return model, player14, ball

def evaluate_model(model, player14, ball, num_runs=3):
    """Model değerlendirme"""
    print(f"\n=== MODEL DEĞERLENDİRMESİ ({num_runs} run) ===")
    
    env = BalancedPlayer14Env(player14, ball)
    all_results = []
    
    for run in range(num_runs):
        print(f"🏃 Run {run+1}/{num_runs}")
        
        obs, _ = env.reset()
        
        # Sonuç kaydetme
        agent_positions = []
        target_positions = []
        rewards = []
        position_errors = []
        movement_similarities = []
        predicted_movements = []
        real_movements = []
        speed_ratios = []
        success_flags = []
        
        test_length = min(len(player14) - 1, 1000)
        total_successes = 0
        
        for step in range(test_length):
            # Model tahmini
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Sonuçları kaydet
            agent_positions.append(info['agent_pos'].copy())
            target_positions.append(info['target_pos'].copy())
            rewards.append(reward)
            position_errors.append(info['position_error'])
            movement_similarities.append(info.get('movement_similarity', 0.0))
            predicted_movements.append(info['predicted_movement'].copy())
            real_movements.append(info['real_movement'].copy())
            speed_ratios.append(info.get('speed_ratio', 1.0))
            success_flags.append(info['success'])
            
            if info['success']:
                total_successes += 1
            
            if terminated or truncated:
                print(f"  Episode {step+1} adımda bitti")
                break
        
        # Bu run'ın sonuçları
        all_results.append({
            'agent_positions': np.array(agent_positions),
            'target_positions': np.array(target_positions),
            'position_errors': np.array(position_errors),
            'rewards': np.array(rewards),
            'movement_similarities': np.array(movement_similarities),
            'predicted_movements': np.array(predicted_movements),
            'real_movements': np.array(real_movements),
            'speed_ratios': np.array(speed_ratios),
            'success_flags': np.array(success_flags),
            'total_successes': total_successes,
            'steps': len(position_errors)
        })
    
    # En iyi run'ı seç
    best_run_idx = np.argmin([np.mean(r['position_errors']) for r in all_results])
    best_run = all_results[best_run_idx]
    
    print(f"📊 En iyi run: #{best_run_idx + 1}")
    
    # Görselleştirme
    create_analysis_plots(best_run, all_results)
    
    # Performans raporu
    print_performance_report(best_run, all_results)
    
    return best_run, all_results

def create_analysis_plots(best_run, all_results):
    """Analiz grafikleri oluştur"""
    
    # En iyi run'ın verileri
    agent_positions = best_run['agent_positions']
    target_positions = best_run['target_positions']
    position_errors = best_run['position_errors']
    rewards = best_run['rewards']
    movement_similarities = best_run['movement_similarities']
    predicted_movements = best_run['predicted_movements']
    real_movements = best_run['real_movements']
    speed_ratios = best_run['speed_ratios']
    success_flags = best_run['success_flags']
    
    # 3x3 plot oluştur
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle('🎯 BALANCED Player 14 Agent - Performans Analizi', fontsize=20, fontweight='bold')
    
    # 1. Yörünge karşılaştırması
    ax1 = axes[0, 0]
    ax1.plot(target_positions[:, 0], target_positions[:, 1], 'b-', 
             linewidth=4, label='Gerçek Player 14', alpha=0.8)
    ax1.plot(agent_positions[:, 0], agent_positions[:, 1], 'r--', 
             linewidth=3, label='Balanced Agent', alpha=0.7)
    
    # Başarı noktalarını vurgula
    success_indices = np.where(success_flags)[0]
    if len(success_indices) > 0:
        ax1.scatter(agent_positions[success_indices, 0], agent_positions[success_indices, 1], 
                   c='gold', s=150, marker='*', label=f'Başarılar ({len(success_indices)})', 
                   zorder=5, edgecolors='black', linewidth=1)
    
    ax1.scatter(target_positions[0, 0], target_positions[0, 1], 
                c='green', s=200, marker='o', label='Start', zorder=6)
    ax1.scatter(target_positions[-1, 0], target_positions[-1, 1], 
                c='red', s=200, marker='X', label='End', zorder=6)
    
    ax1.set_title('🎯 YÖRÜNGE KARŞILAŞTIRMASI', fontweight='bold')
    ax1.set_xlabel('X Pozisyonu')
    ax1.set_ylabel('Y Pozisyonu')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Pozisyon hatası
    ax2 = axes[0, 1]
    ax2.plot(position_errors, 'red', alpha=0.6, linewidth=1, label='Ham Hata')
    
    # Moving average
    if len(position_errors) > 20:
        moving_avg = np.convolve(position_errors, np.ones(20)/20, mode='valid')
        ax2.plot(range(19, len(position_errors)), moving_avg, 'darkred', 
                linewidth=3, label='20-step MA', alpha=0.9)
    
    ax2.axhline(y=np.mean(position_errors), color='blue', linestyle='--', 
                linewidth=2, label=f'Ortalama: {np.mean(position_errors):.2f}')
    ax2.axhline(y=1.0, color='gold', linestyle='-', 
                linewidth=2, label='Başarı Eşiği')
    
    ax2.set_title('📈 POZİSYON HATASI', fontweight='bold')
    ax2.set_xlabel('Zaman Adımı')
    ax2.set_ylabel('Hata')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Hız karşılaştırması
    ax3 = axes[0, 2]
    pred_speeds = np.linalg.norm(predicted_movements, axis=1)
    real_speeds = np.linalg.norm(real_movements, axis=1)
    
    ax3.plot(real_speeds, 'b-', alpha=0.7, linewidth=2, label='Gerçek Hız')
    ax3.plot(pred_speeds, 'r-', alpha=0.7, linewidth=2, label='Agent Hız')
    
    # Moving averages
    if len(pred_speeds) > 20:
        real_smooth = np.convolve(real_speeds, np.ones(20)/20, mode='valid')
        pred_smooth = np.convolve(pred_speeds, np.ones(20)/20, mode='valid')
        ax3.plot(range(19, len(real_speeds)), real_smooth, 'darkblue', 
                linewidth=3, label='Gerçek (smooth)', alpha=0.9)
        ax3.plot(range(19, len(pred_speeds)), pred_smooth, 'darkred', 
                linewidth=3, label='Agent (smooth)', alpha=0.9)
    
    ax3.axhline(y=np.mean(real_speeds), color='blue', linestyle=':', alpha=0.8)
    ax3.axhline(y=np.mean(pred_speeds), color='red', linestyle=':', alpha=0.8)
    
    ax3.set_title('🚀 HIZ KARŞILAŞTIRMASI', fontweight='bold')
    ax3.set_xlabel('Zaman Adımı')
    ax3.set_ylabel('Hız')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Hız oranı analizi
    ax4 = axes[1, 0]
    ax4.plot(speed_ratios, 'purple', linewidth=2, alpha=0.8, label='Hız Oranı')
    
    # İdeal bölgeyi vurgula
    ax4.axhspan(0.8, 1.2, alpha=0.2, color='green', label='İdeal Bölge')
    ax4.axhspan(0.6, 1.5, alpha=0.1, color='yellow', label='Kabul Edilebilir')
    
    ax4.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, label='Perfect Match')
    ax4.axhline(y=np.mean(speed_ratios), color='red', linestyle='--', 
                linewidth=2, label=f'Ortalama: {np.mean(speed_ratios):.2f}')
    
    ax4.set_title('⚡ HIZ ORANI ANALİZİ', fontweight='bold')
    ax4.set_xlabel('Zaman Adımı')
    ax4.set_ylabel('Agent/Gerçek Hız Oranı')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Hareket benzerliği
    ax5 = axes[1, 1]
    ax5.plot(movement_similarities, 'green', linewidth=2, alpha=0.8)
    
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax5.axhline(y=np.mean(movement_similarities), color='red', linestyle='--', 
                linewidth=2, label=f'Ortalama: {np.mean(movement_similarities):.3f}')
    
    # Pozitif/negatif alanları
    ax5.fill_between(range(len(movement_similarities)), 0, movement_similarities, 
                     where=(movement_similarities >= 0), color='green', alpha=0.2)
    ax5.fill_between(range(len(movement_similarities)), 0, movement_similarities, 
                     where=(movement_similarities < 0), color='red', alpha=0.2)
    
    ax5.set_title('🎯 HAREKET YÖN BENZERLİĞİ', fontweight='bold')
    ax5.set_xlabel('Zaman Adımı')
    ax5.set_ylabel('Kosinüs Benzerliği')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Reward evrimi
    ax6 = axes[1, 2]
    ax6.plot(rewards, 'orange', alpha=0.6, linewidth=1, label='Ham Reward')
    
    # Pozitif/negatif reward'ları ayır
    positive_rewards = np.where(rewards >= 0, rewards, 0)
    negative_rewards = np.where(rewards < 0, rewards, 0)
    
    ax6.fill_between(range(len(rewards)), 0, positive_rewards, 
                     color='green', alpha=0.3, label='Pozitif')
    ax6.fill_between(range(len(rewards)), 0, negative_rewards, 
                     color='red', alpha=0.3, label='Negatif')
    
    if len(rewards) > 25:
        reward_smooth = np.convolve(rewards, np.ones(25)/25, mode='valid')
        ax6.plot(range(24, len(rewards)), reward_smooth, 'darkorange', 
                linewidth=3, label='25-step MA', alpha=0.9)
    
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax6.axhline(y=np.mean(rewards), color='blue', linestyle='--', 
                linewidth=2, label=f'Ortalama: {np.mean(rewards):.1f}')
    
    ax6.set_title('🎁 REWARD EVRİMİ', fontweight='bold')
    ax6.set_xlabel('Zaman Adımı')
    ax6.set_ylabel('Reward')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Başarı oranları
    ax7 = axes[2, 0]
    thresholds = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    success_rates = []
    
    for threshold in thresholds:
        success_rate = (position_errors < threshold).mean() * 100
        success_rates.append(success_rate)
    
    colors = plt.cm.RdYlGn([rate/100 for rate in success_rates])
    bars = ax7.bar(range(len(thresholds)), success_rates, color=colors, alpha=0.8)
    
    ax7.set_xticks(range(len(thresholds)))
    ax7.set_xticklabels([f'<{t}' for t in thresholds])
    ax7.set_title('✅ BAŞARI ORANLARI', fontweight='bold')
    ax7.set_xlabel('Hata Threshold')
    ax7.set_ylabel('Başarı Oranı (%)')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Bar değerlerini ekle
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 8. Mesafe verimliliği
    ax8 = axes[2, 1]
    
    # Toplam mesafe karşılaştırması
    real_total_distance = np.sum(np.linalg.norm(real_movements, axis=1))
    agent_total_distance = np.sum(np.linalg.norm(predicted_movements, axis=1))
    distance_efficiency = real_total_distance / max(agent_total_distance, 1e-6)
    
    # Kümülatif mesafe
    real_cumulative = np.cumsum(np.linalg.norm(real_movements, axis=1))
    agent_cumulative = np.cumsum(np.linalg.norm(predicted_movements, axis=1))
    
    ax8.plot(real_cumulative, 'b-', linewidth=3, label='Gerçek Kümülatif')
    ax8.plot(agent_cumulative, 'r--', linewidth=3, label='Agent Kümülatif')
    
    # Efficiency line
    if len(real_cumulative) > 0:
        perfect_line = agent_cumulative * distance_efficiency
        ax8.plot(perfect_line, 'g:', linewidth=2, alpha=0.7, label='Mükemmel Efficiency')
    
    ax8.set_title(f'📏 MESAFE VERİMLİLİĞİ\nEfficiency: {distance_efficiency:.2f}', fontweight='bold')
    ax8.set_xlabel('Zaman Adımı')
    ax8.set_ylabel('Kümülatif Mesafe')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Çoklu run karşılaştırması
    ax9 = axes[2, 2]
    
    run_errors = [np.mean(r['position_errors']) for r in all_results]
    run_successes = [r['total_successes']/r['steps']*100 for r in all_results]
    run_similarities = [np.mean(r['movement_similarities']) for r in all_results]
    
    x = np.arange(len(all_results))
    width = 0.25
    
    bars1 = ax9.bar(x - width, run_errors, width, label='Ortalama Hata', 
                   color='red', alpha=0.7)
    bars2 = ax9.bar(x, run_successes, width, label='Başarı Oranı (%)', 
                   color='green', alpha=0.7)
    bars3 = ax9.bar(x + width, [(s+1)*25 for s in run_similarities], width, 
                   label='Benzerlik*25', color='blue', alpha=0.7)
    
    ax9.set_xlabel('Run Numarası')
    ax9.set_ylabel('Değer')
    ax9.set_title('🔄 ÇOKLU RUN KARŞILAŞTIRMASI', fontweight='bold')
    ax9.set_xticks(x)
    ax9.set_xticklabels([f'Run {i+1}' for i in range(len(all_results))])
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('balanced_player14_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_performance_report(best_run, all_results):
    """Performans raporu yazdır"""
    print("\n" + "="*80)
    print("🎯 BALANCED PLAYER 14 AGENT PERFORMANS RAPORU")
    print("="*80)
    
    position_errors = best_run['position_errors']
    speed_ratios = best_run['speed_ratios']
    movement_similarities = best_run['movement_similarities']
    predicted_movements = best_run['predicted_movements']
    real_movements = best_run['real_movements']
    rewards = best_run['rewards']
    total_successes = best_run['total_successes']
    
    # Genel metrikler
    print(f"📊 GENEL PERFORMANS:")
    print(f"   • Toplam adım: {len(position_errors)}")
    print(f"   • Ortalama pozisyon hatası: {np.mean(position_errors):.3f}")
    print(f"   • Medyan pozisyon hatası: {np.median(position_errors):.3f}")
    print(f"   • Min hata: {np.min(position_errors):.3f}")
    print(f"   • Max hata: {np.max(position_errors):.3f}")
    print(f"   • Başarı sayısı: {total_successes}")
    print(f"   • Başarı oranı: {(total_successes / len(position_errors) * 100):.1f}%")
    
    # Hız analizi
    avg_real_speed = np.mean(np.linalg.norm(real_movements, axis=1))
    avg_agent_speed = np.mean(np.linalg.norm(predicted_movements, axis=1))
    avg_speed_ratio = np.mean(speed_ratios)
    
    print(f"\n🚀 HIZ ANALİZİ:")
    print(f"   • Ortalama gerçek hız: {avg_real_speed:.3f}")
    print(f"   • Ortalama agent hız: {avg_agent_speed:.3f}")
    print(f"   • Ortalama hız oranı: {avg_speed_ratio:.3f}")
    print(f"   • İdeal hız oranı (0.8-1.2): {((speed_ratios >= 0.8) & (speed_ratios <= 1.2)).mean() * 100:.1f}%")
    print(f"   • Aşırı hız (>2.0): {(speed_ratios > 2.0).mean() * 100:.1f}%")
    
    # Mesafe verimliliği
    real_total = np.sum(np.linalg.norm(real_movements, axis=1))
    agent_total = np.sum(np.linalg.norm(predicted_movements, axis=1))
    efficiency = real_total / max(agent_total, 1e-6)
    
    print(f"\n📏 MESAFE VERİMLİLİĞİ:")
    print(f"   • Gerçek toplam mesafe: {real_total:.2f}")
    print(f"   • Agent toplam mesafe: {agent_total:.2f}")
    print(f"   • Verimlilik oranı: {efficiency:.3f}")
    print(f"   • Mesafe fazlası: {((agent_total/max(real_total, 1e-6) - 1) * 100):.1f}%")
    
    # Hareket benzerliği
    print(f"\n🎯 HAREKET BENZERLİĞİ:")
    print(f"   • Ortalama benzerlik: {np.mean(movement_similarities):.3f}")
    print(f"   • Pozitif benzerlik oranı: {(movement_similarities > 0).mean() * 100:.1f}%")
    print(f"   • Yüksek benzerlik (>0.5): {(movement_similarities > 0.5).mean() * 100:.1f}%")
    
    # Reward analizi
    print(f"\n🎁 REWARD ANALİZİ:")
    print(f"   • Ortalama reward: {np.mean(rewards):.2f}")
    print(f"   • Toplam reward: {np.sum(rewards):.2f}")
    print(f"   • Pozitif reward oranı: {(rewards > 0).mean() * 100:.1f}%")
    print(f"   • Max reward: {np.max(rewards):.2f}")
    print(f"   • Min reward: {np.min(rewards):.2f}")
    
    # Başarı threshold analizi
    print(f"\n✅ BAŞARI ANALİZİ:")
    thresholds = [0.5, 1.0, 2.0, 3.0, 5.0]
    for threshold in thresholds:
        success_rate = (position_errors < threshold).mean() * 100
        count = np.sum(position_errors < threshold)
        print(f"   • <{threshold:.1f} birim: {success_rate:.1f}% ({count}/{len(position_errors)})")
    
    # Çoklu run özeti
    if len(all_results) > 1:
        all_errors = [np.mean(r['position_errors']) for r in all_results]
        all_speeds = [np.mean(r['speed_ratios']) for r in all_results]
        all_sims = [np.mean(r['movement_similarities']) for r in all_results]
        
        print(f"\n🔄 ÇOKLU RUN ÖZETİ:")
        print(f"   • Ortalama hata: {np.mean(all_errors):.3f} ± {np.std(all_errors):.3f}")
        print(f"   • Ortalama hız oranı: {np.mean(all_speeds):.3f} ± {np.std(all_speeds):.3f}")
        print(f"   • Ortalama benzerlik: {np.mean(all_sims):.3f} ± {np.std(all_sims):.3f}")
    
    # Genel değerlendirme
    score = calculate_overall_score(best_run)
    print(f"\n🏆 GENEL SKOR: {score:.1f}/100")
    
    if score >= 85:
        grade = "🥇 MÜKEMMEL"
    elif score >= 75:
        grade = "🥈 ÇOK İYİ" 
    elif score >= 65:
        grade = "🥉 İYİ"
    elif score >= 50:
        grade = "📈 ORTA"
    else:
        grade = "📉 GELİŞTİRİLMELİ"
    
    print(f"🎯 PERFORMANS: {grade}")
    print("="*80)

def calculate_overall_score(result):
    """Genel performans skoru hesapla"""
    position_errors = result['position_errors']
    speed_ratios = result['speed_ratios']
    movement_similarities = result['movement_similarities']
    success_rate = result['total_successes'] / result['steps']
    
    # Pozisyon doğruluğu (40%)
    pos_score = max(0, 100 - np.mean(position_errors) * 20) * 0.4
    
    # Hız uyumu (30%)
    ideal_speed_ratio = ((speed_ratios >= 0.8) & (speed_ratios <= 1.2)).mean()
    speed_score = ideal_speed_ratio * 100 * 0.3
    
    # Hareket benzerliği (20%)
    sim_score = ((np.mean(movement_similarities) + 1) / 2 * 100) * 0.2
    
    # Başarı oranı (10%)
    success_score = success_rate * 100 * 0.1
    
    return pos_score + speed_score + sim_score + success_score

def load_trained_model():
    """Eğitilmiş modeli yükle"""
    try:
        model = PPO.load("balanced_player14_model")
        print("✅ Model başarıyla yüklendi: balanced_player14_model.zip")
        return model
    except:
        print("❌ Model yüklenemedi! Önce eğitim yapmanız gerekiyor.")
        return None

def main():
    """Ana fonksiyon - Tek mod"""
    print("🎯 BALANCED PLAYER 14 TRACKING SYSTEM")
    print("="*50)
    
    # Timesteps ayarlama
    timesteps = input("Eğitim timesteps (default: 200000): ").strip()
    if timesteps.isdigit():
        timesteps = int(timesteps)
    else:
        timesteps = 200000
    
    print(f"\n🚀 Eğitim başlıyor ({timesteps} timesteps)...")
    
    try:
        # Eğitim
        model, player14, ball = train_balanced_agent(timesteps)
        
        if model is not None:
            print("\n📊 Eğitim tamamlandı, değerlendirme başlıyor...")
            
            # Değerlendirme
            best_run, all_results = evaluate_model(model, player14, ball, num_runs=3)
            
            # Model ve metadata kaydetme
            metadata = {
                'timesteps': timesteps,
                'best_score': calculate_overall_score(best_run),
                'avg_position_error': float(np.mean(best_run['position_errors'])),
                'avg_speed_ratio': float(np.mean(best_run['speed_ratios'])),
                'success_rate': float(best_run['total_successes'] / best_run['steps']),
                'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            import json
            with open('balanced_player14_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print("✅ Tüm işlemler tamamlandı!")
            print("   • balanced_player14_model.zip")
            print("   • balanced_player14_metadata.json")
            print("   • balanced_player14_analysis.png")
        
    except KeyboardInterrupt:
        print("\n⚠️ İşlem kullanıcı tarafından iptal edildi.")
    except Exception as e:
        print(f"\n❌ Hata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()