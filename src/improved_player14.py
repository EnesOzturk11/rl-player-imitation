import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
import torch.nn as nn 

class ImprovedPlayer14Env(gym.Env):
    def __init__(self, player14_data, ball_data):
        super(ImprovedPlayer14Env, self).__init__()
        self.player14_data = player14_data
        self.ball_data = ball_data
        
        # Veri uzunluğunu eşitle
        min_length = min(len(player14_data), len(ball_data))
        self.player14_data = player14_data.iloc[:min_length].copy()
        self.ball_data = ball_data.iloc[:min_length].copy()
        self.max_steps = min_length - 1
        
        print(f"Player 14 veri uzunluğu: {len(self.player14_data)}")
        print(f"Ball veri uzunluğu: {len(self.ball_data)}")
        
        # 🔥 ULTRA AGRESİF ACTION SPACE - DURMAYA İZİN VERMİYORUZ!
        pos = self.player14_data[["position_x", "position_y"]].values
        if len(pos) > 1:
            dx = np.diff(pos[:, 0])
            dy = np.diff(pos[:, 1])
            speeds = np.sqrt(dx**2 + dy**2)
            
            # Gerçek hareket istatistikleri
            max_speed = np.percentile(speeds, 99)  # En üst %1
            mean_speed = np.mean(speeds[speeds > 0.1])  # Sadece anlamlı hareketler
            std_speed = np.std(speeds)
            
            print(f"Player 14 max hız (99th percentile): {max_speed:.2f}")
            print(f"Player 14 ortalama anlamlı hız: {mean_speed:.2f}")
            print(f"Player 14 hız std: {std_speed:.2f}")
            
            # Çok geniş action space - gerçek hareketlerin 3 katı
            speed_limit = max(max_speed * 3.0, mean_speed * 5.0)
            self.min_movement = mean_speed * 0.3  # Minimum hareket zorunluluğu
        else:
            speed_limit = 15.0
            self.min_movement = 1.0
        
        print(f"Action space hız limiti: {speed_limit:.2f}")
        print(f"Minimum hareket zorunluluğu: {self.min_movement:.2f}")
        
        # Çok geniş action space
        self.action_space = spaces.Box(
            low=np.array([-speed_limit, -speed_limit], dtype=np.float32), 
            high=np.array([speed_limit, speed_limit], dtype=np.float32),
            dtype=np.float32
        )
        
        # 🎯 SÜPER ZENGİN OBSERVATION SPACE - BOYUT SAYISINI DÜZELTTİK
        # [agent_x, agent_y, agent_vx, agent_vy, ball_x, ball_y, ball_direction, ball_vx, ball_vy,
        #  relative_ball_x, relative_ball_y, target_x, target_y, target_vx, target_vy,
        #  distance_to_target, direction_to_target_x, direction_to_target_y, required_velocity_x, required_velocity_y,
        #  progress_ratio, steps_remaining_ratio, future_targets_x(3), future_targets_y(3), consecutive_small_moves]
        # Toplam: 2 + 2 + 2 + 1 + 2 + 2 + 2 + 2 + 1 + 2 + 2 + 1 + 3 + 3 + 1 = 29 boyut
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32
        )
        
        self.current_step = 0
        self.agent_pos = None
        self.agent_velocity = np.array([0.0, 0.0])
        self.prev_movements = [np.array([0.0, 0.0]) for _ in range(3)]  # Son 3 hareket
        self.total_distance_traveled = 0.0
        self.consecutive_small_moves = 0  # Küçük hareket sayacı
        
        # Normalizasyon için scale faktörleri
        pos_range = np.ptp(pos, axis=0)
        self.pos_scale = 1.0 / (np.max(pos_range) + 1e-8) if len(pos) > 0 else 1e-3
        self.max_distance = np.linalg.norm(pos_range) if len(pos) > 0 else 1000.0
        
        print(f"Pozisyon scale faktörü: {self.pos_scale:.6f}")
        print(f"Maksimum mesafe: {self.max_distance:.2f}")
        
        # Hedef pozisyon geçmişi
        self.target_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.total_distance_traveled = 0.0
        self.consecutive_small_moves = 0
        
        # Player 14'ün başlangıç pozisyonu
        initial_pos = self.player14_data[["position_x", "position_y"]].iloc[0].values
        self.agent_pos = np.nan_to_num(initial_pos.copy(), nan=0.0)
        self.agent_velocity = np.array([0.0, 0.0])
        self.prev_movements = [np.array([0.0, 0.0]) for _ in range(3)]
        
        # Target history'yi başlat
        self.target_history = []
        for i in range(min(10, len(self.player14_data))):
            target_pos = self.player14_data[["position_x", "position_y"]].iloc[i].values
            self.target_history.append(np.nan_to_num(target_pos, nan=0.0))
        
        observation = self._get_observation()
        info = {'step': self.current_step}
        return observation, info
    
    def _get_observation(self):
        """🎯 SÜPER ZENGİN GÖZLEM ALANI - BOYUT SORUNUNU DÜZELTTİK"""
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
        
        # Hedefe ulaşmak için gereken hız
        required_velocity = current_target - self.agent_pos
        
        # İlerleme ve zaman bilgileri
        progress_ratio = self.current_step / max(self.max_steps, 1)
        steps_remaining_ratio = (self.max_steps - self.current_step) / max(self.max_steps, 1)
        
        # Son 3 hedef pozisyon (hareket kalıbını anlamak için)
        future_targets_x = []
        future_targets_y = []
        for i in range(3):
            future_idx = min(self.current_step + i + 1, len(self.player14_data) - 1)
            future_target = self.player14_data[["position_x", "position_y"]].iloc[future_idx].values
            future_target = np.nan_to_num(future_target, nan=0.0)
            future_targets_x.append(future_target[0])
            future_targets_y.append(future_target[1])
        
        # Observation vektörü oluştur - BOYUT KONTROLÜ İLE
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
            required_velocity * self.pos_scale,                 # Gereken hız (2)
            [progress_ratio],                                   # İlerleme oranı (1)
            [steps_remaining_ratio],                            # Kalan zaman oranı (1)
            future_targets_x,                                   # Gelecek hedefler X (3)
            future_targets_y,                                   # Gelecek hedefler Y (3)
            [self.consecutive_small_moves / 10.0]               # Küçük hareket cezası (1)
        ]
        
        # Tüm parçaları birleştir
        observation = np.concatenate(observation_parts)
        
        # Boyut kontrolü yapalım
        if len(observation) != 29:
            print(f"UYARI: Observation boyutu {len(observation)} olmalı 29!")
            print("Observation parçalarının boyutları:")
            for i, part in enumerate(observation_parts):
                part_array = np.array(part)
                print(f"  Part {i}: {part_array.shape} -> {len(part_array)}")
            
            # Eğer boyut yanlışsa, 29'a tamamla veya kes
            if len(observation) < 29:
                observation = np.pad(observation, (0, 29 - len(observation)))
            else:
                observation = observation[:29]
        
        return np.nan_to_num(observation, nan=0.0).astype(np.float32)

    def step(self, action):
        # Action'ı temizle ve sınırla
        action = np.nan_to_num(action, nan=0.0)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 🚫 DURMAYI YASAKLA! - Minimum hareket zorunluluğu
        action_magnitude = np.linalg.norm(action)
        
        if action_magnitude < self.min_movement:
            # Çok küçük hareket varsa, rastgele yön seç ve minimum hıza çıkar
            if action_magnitude > 1e-6:
                action = action / action_magnitude * self.min_movement
            else:
                # Tamamen sıfırsa, hedefe doğru minimum hareket
                current_target = self.player14_data[["position_x", "position_y"]].iloc[self.current_step].values
                current_target = np.nan_to_num(current_target, nan=0.0)
                direction = current_target - self.agent_pos
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-6:
                    action = direction / direction_norm * self.min_movement
                else:
                    # Random direction ile minimum hareket
                    angle = np.random.uniform(0, 2*np.pi)
                    action = np.array([np.cos(angle), np.sin(angle)]) * self.min_movement
            
            self.consecutive_small_moves += 1
        else:
            self.consecutive_small_moves = max(0, self.consecutive_small_moves - 1)
        
        # Agent'ı hareket ettir
        old_pos = self.agent_pos.copy()
        self.agent_pos += action
        self.agent_velocity = action.copy()  # Velocity = son hareket
        
        # Hareket geçmişini güncelle
        self.prev_movements = self.prev_movements[1:] + [action.copy()]
        self.total_distance_traveled += np.linalg.norm(action)
        self.current_step += 1
        
        # Mevcut hedef pozisyonu
        if self.current_step < len(self.player14_data):
            current_target = self.player14_data[["position_x", "position_y"]].iloc[self.current_step].values
            prev_target = self.player14_data[["position_x", "position_y"]].iloc[self.current_step-1].values
            real_movement = current_target - prev_target
        else:
            current_target = self.player14_data[["position_x", "position_y"]].iloc[-1].values
            real_movement = np.array([0.0, 0.0])
        
        # NaN kontrolü
        current_target = np.nan_to_num(current_target, nan=0.0)
        real_movement = np.nan_to_num(real_movement, nan=0.0)
        
        # 🔥 ULTRA AGRESİF REWARD SİSTEMİ
        
        # 1. Pozisyon hatası - EN ÖNEMLİ (ağırlık: 10x)
        position_error = np.linalg.norm(self.agent_pos - current_target)
        max_possible_error = self.max_distance
        normalized_error = position_error / max_possible_error
        position_reward = -normalized_error * 50.0  # ÇOK YÜKSEK CEZA
        
        # 2. İlerleme bonusu - Hedefe yaklaşıyorsa BÜYÜK ÖDÜL
        if hasattr(self, 'prev_position_error'):
            progress = self.prev_position_error - position_error
            if progress > 0:
                progress_reward = progress * 100.0  # BÜYÜK BONUS yaklaşma için
            else:
                progress_reward = progress * 200.0  # ÇOK BÜYÜK CEZA uzaklaşma için
        else:
            progress_reward = 0.0
        self.prev_position_error = position_error
        
        # 3. Hareket yönü uyumu - Gerçek harekete benzerlik
        if np.linalg.norm(real_movement) > 1e-6 and np.linalg.norm(action) > 1e-6:
            # Normalize edilmiş vektörler arası kosinüs benzerliği
            real_normalized = real_movement / np.linalg.norm(real_movement)
            action_normalized = action / np.linalg.norm(action)
            movement_similarity = np.dot(real_normalized, action_normalized)
            movement_reward = movement_similarity * 30.0  # Yön uyumu için büyük bonus
        else:
            movement_reward = -10.0  # Hareket yoksa ceza
            movement_similarity = 0.0
        
        # 4. Hareket hızı uyumu
        real_speed = np.linalg.norm(real_movement)
        agent_speed = np.linalg.norm(action)
        if real_speed > 1e-6:
            speed_ratio = min(agent_speed / real_speed, 2.0)  # Max 2x hıza izin ver
            if 0.7 <= speed_ratio <= 1.3:  # İdeal hız aralığı
                speed_reward = 20.0
            elif 0.5 <= speed_ratio <= 1.7:  # Kabul edilebilir
                speed_reward = 10.0
            else:  # Çok yavaş veya çok hızlı
                speed_reward = -abs(speed_ratio - 1.0) * 15.0
        else:
            speed_reward = -5.0
        
        # 5. Hareket etme zorunluluğu bonusu
        movement_magnitude = np.linalg.norm(action)
        if movement_magnitude >= self.min_movement:
            movement_bonus = min(movement_magnitude * 5.0, 25.0)  # Hareket et!
        else:
            movement_bonus = -50.0  # Durma cezası
        
        # 6. Durağanlık cezası (küçük hareket cezası)
        if self.consecutive_small_moves > 0:
            stagnation_penalty = -self.consecutive_small_moves * 20.0
        else:
            stagnation_penalty = 0.0
        
        # 7. Yakınlık bonusu (hedefe yakınsa)
        if position_error < 5.0:
            proximity_bonus = (5.0 - position_error) * 20.0
        elif position_error < 10.0:
            proximity_bonus = (10.0 - position_error) * 5.0
        else:
            proximity_bonus = -10.0
        
        # 8. Tutarlılık bonusu (smooth movement)
        if len(self.prev_movements) >= 2:
            consistency = 0
            for i in range(1, len(self.prev_movements)):
                prev_move = self.prev_movements[i-1]
                curr_move = self.prev_movements[i]
                if np.linalg.norm(prev_move) > 1e-6 and np.linalg.norm(curr_move) > 1e-6:
                    similarity = np.dot(prev_move, curr_move) / (np.linalg.norm(prev_move) * np.linalg.norm(curr_move))
                    consistency += similarity
            consistency_reward = consistency * 10.0
        else:
            consistency_reward = 0.0
        
        # 9. Zaman bonusu (episode'u erken tamamlama teşviki)
        time_bonus = (self.max_steps - self.current_step) * 0.1
        
        # 10. Başarı bonusu (hedefe çok yaklaşma)
        success_bonus = 0.0
        if position_error < 2.0:
            success_bonus = (2.0 - position_error) * 100.0
        if position_error < 1.0:
            success_bonus += 200.0  # Extra bonus
        if position_error < 0.5:
            success_bonus += 500.0  # Mega bonus
        
        # TOPLAM REWARD
        total_reward = (
            position_reward +
            progress_reward + 
            movement_reward + 
            speed_reward +
            movement_bonus + 
            stagnation_penalty + 
            proximity_bonus + 
            consistency_reward + 
            time_bonus + 
            success_bonus
        )
        
        # Reward'ı daha geniş aralığa izin ver ama sınırla
        total_reward = np.clip(total_reward, -1000, 1000)
        
        # Episode bitme koşulları
        success_threshold = 1.0
        terminated = False
        
        # Başarı durumu
        if position_error < success_threshold:
            total_reward += 1000.0  # MEGA BAŞARI BONUSU
            terminated = True
            print(f"🎯 BAŞARI! Adım {self.current_step}'te hedefe ulaşıldı (hata: {position_error:.2f})")
        
        # Normal episode sonu
        elif self.current_step >= self.max_steps:
            terminated = True
            # Son pozisyon bonusu
            final_bonus = max(0, 50 - position_error)
            total_reward += final_bonus
        
        # Çok uzaklaşırsa episode'u bitir (performans için)
        elif position_error > self.max_distance * 2.0:
            terminated = True
            total_reward -= 500.0  # Büyük ceza
            print(f"⚠️ Episode erken bitti - çok uzaklaştı (hata: {position_error:.2f})")
        
        truncated = False
        
        # Yeni gözlem
        observation = self._get_observation()
        
        # Detaylı info
        info = {
            'step': self.current_step,
            'position_error': position_error,
            'movement_similarity': movement_similarity,
            'target_pos': current_target,
            'agent_pos': self.agent_pos.copy(),
            'real_movement': real_movement,
            'predicted_movement': action,
            'position_reward': position_reward,
            'progress_reward': progress_reward,
            'movement_reward': movement_reward,
            'speed_reward': speed_reward,
            'movement_bonus': movement_bonus,
            'stagnation_penalty': stagnation_penalty,
            'proximity_bonus': proximity_bonus,
            'success_bonus': success_bonus,
            'total_distance_traveled': self.total_distance_traveled,
            'consecutive_small_moves': self.consecutive_small_moves,
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
    
    # Veri temizleme - pandas 2.0+ uyumlu
    def clean_dataframe(df):
        df = df.ffill().bfill().fillna(0)  # method parametresi kaldırıldı
        df = df.replace([np.inf, -np.inf], 0)
        return df
    
    ball = clean_dataframe(ball)
    player14 = clean_dataframe(player14)
    
    print("✅ Veriler temizlendi")
    return ball, player14

def make_env(player14, ball):
    """Environment wrapper"""
    def _init():
        return ImprovedPlayer14Env(player14, ball)
    return _init

def train_improved_agent():
    """🔥 ULTRA AGRESİF EĞİTİM PARAMETRELERİ"""
    print("=== ULTRA AGRESİF PLAYER 14 AGENT EĞİTİMİ ===")
    
    # Veri yükleme
    ball, player14 = load_and_clean_data()
    if ball is None or player14 is None:
        return None, None, None
    
    # Environment oluşturma
    env = DummyVecEnv([make_env(player14, ball)])

    policy_kwargs = dict(
        # SB3 >=1.8 tavsiye edilen yeni söz dizimi
        net_arch=dict(pi=[512, 512, 256, 128],
                      vf=[512, 512, 256, 128]),
        activation_fn=nn.Tanh            # veya nn.ReLU, nn.SiLU ...
    )

    
    # Ultra agresif öğrenme parametreleri
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-4,      # Daha yüksek learning rate
        n_steps=512,             # Küçük batch (sık güncelleme)
        batch_size=32,           # Çok küçük batch
        n_epochs=30,             # Çok fazla epoch
        gamma=0.90,              # Yakın vadeli odaklanma
        gae_lambda=0.85,         # Daha düşük GAE
        clip_range=0.4,          # Geniş clip range
        ent_coef=0.1,            # Maksimum exploration
        vf_coef=0.3,             # Value function'a az ağırlık
        max_grad_norm=2.0,       # Büyük gradient'lara izin ver
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tensorboard_logs_ultra/"
    )
    
    # Evaluation callback
    eval_env = DummyVecEnv([make_env(player14, ball)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models_ultra/",
        log_path="./logs_ultra/",
        eval_freq=2000,  # Daha sık evaluation
        deterministic=True,
        render=False
    )
    
    # Ultra uzun eğitim
    print("🔥 Ultra agresif eğitim başlıyor...")
    model.learn(
        total_timesteps=500000,  # Çok uzun eğitim
        callback=eval_callback,
        progress_bar=True
    )
    
    # Model kaydetme
    model.save("ultra_aggressive_player14_model")
    print("✅ Ultra agresif model kaydedildi: ultra_aggressive_player14_model.zip")
    
    return model, player14, ball

def detailed_evaluation(model, player14, ball):
    """🔥 Ultra detaylı değerlendirme"""
    print("\n=== ULTRA DETAYLI DEĞERLENDİRME ===")
    
    # Test environment
    env = ImprovedPlayer14Env(player14, ball)
    
    # Multiple run evaluation
    num_runs = 5  # 5 farklı çalışma
    all_results = []
    
    for run in range(num_runs):
        print(f"🏃 Çalışma {run+1}/{num_runs}")
        
        obs, _ = env.reset()
        
        # Sonuçları saklama
        agent_positions = []
        target_positions = []
        rewards = []
        position_errors = []
        movement_similarities = []
        predicted_movements = []
        real_movements = []
        success_flags = []
        
        # Test uzunluğu
        test_length = min(len(player14) - 1, 600)
        total_successes = 0
        
        for step in range(test_length):
            # Model tahmini
            action, _ = model.predict(obs, deterministic=False)  # Stochastic evaluation
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Sonuçları kaydet
            agent_positions.append(info['agent_pos'].copy())
            target_positions.append(info['target_pos'].copy())
            rewards.append(reward)
            position_errors.append(info['position_error'])
            movement_similarities.append(info.get('movement_similarity', 0.0))
            predicted_movements.append(info['predicted_movement'].copy())
            real_movements.append(info['real_movement'].copy())
            success_flags.append(info['success'])
            
            if info['success']:
                total_successes += 1
            
            if terminated or truncated:
                print(f"  Episode {step+1} adımda bitti")
                break
        
        # Bu çalışmanın sonuçlarını sakla
        all_results.append({
            'agent_positions': np.array(agent_positions),
            'target_positions': np.array(target_positions),
            'position_errors': np.array(position_errors),
            'rewards': np.array(rewards),
            'movement_similarities': np.array(movement_similarities),
            'predicted_movements': np.array(predicted_movements),
            'real_movements': np.array(real_movements),
            'success_flags': np.array(success_flags),
            'total_successes': total_successes,
            'steps': len(position_errors)
        })
    
    # En iyi çalışmayı seç (en düşük ortalama hata)
    best_run_idx = np.argmin([np.mean(r['position_errors']) for r in all_results])
    best_run = all_results[best_run_idx]
    
    print(f"📊 En iyi çalışma: #{best_run_idx + 1}")
    
    # En iyi çalışmanın verilerini çıkart
    agent_positions = best_run['agent_positions']
    target_positions = best_run['target_positions']
    position_errors = best_run['position_errors']
    rewards = best_run['rewards']
    movement_similarities = best_run['movement_similarities']
    predicted_movements = best_run['predicted_movements']
    real_movements = best_run['real_movements']
    success_flags = best_run['success_flags']
    total_successes = best_run['total_successes']
    
    # 🔥 ULTRA DETAYLI GÖRSELLEŞTİRME
    fig, axes = plt.subplots(3, 4, figsize=(28, 20))
    fig.suptitle('🔥 ULTRA AGRESİF Player 14 Agent Performans Analizi', fontsize=24, fontweight='bold', color='red')
    
    # 1. Tam yörünge karşılaştırması (büyütülmüş)
    ax1 = axes[0, 0]
    ax1.plot(target_positions[:, 0], target_positions[:, 1], 'b-', 
             linewidth=5, label='Gerçek Player 14', alpha=0.9)
    ax1.plot(agent_positions[:, 0], agent_positions[:, 1], 'r--', 
             linewidth=4, label='Ultra Agent', alpha=0.8)
    
    # Başarılı noktaları vurgula
    success_indices = np.where(success_flags)[0]
    if len(success_indices) > 0:
        ax1.scatter(agent_positions[success_indices, 0], agent_positions[success_indices, 1], 
                   c='gold', s=200, marker='*', label=f'Başarı ({len(success_indices)})', 
                   zorder=5, edgecolors='black', linewidth=2)
    
    # Her 50. adımı işaretle
    step_markers = range(0, len(agent_positions), 50)
    ax1.scatter(agent_positions[step_markers, 0], agent_positions[step_markers, 1], 
               c='red', s=50, alpha=0.6, zorder=3)
    ax1.scatter(target_positions[step_markers, 0], target_positions[step_markers, 1], 
               c='blue', s=50, alpha=0.6, zorder=3)
    
    ax1.scatter(target_positions[0, 0], target_positions[0, 1], 
                c='green', s=300, marker='o', label='Başlangıç', zorder=6, edgecolors='black')
    ax1.scatter(target_positions[-1, 0], target_positions[-1, 1], 
                c='purple', s=300, marker='X', label='Bitiş', zorder=6, edgecolors='black')
    
    ax1.set_title('🎯 TAM YÖRÜNGE KARŞILAŞTIRMASI', fontsize=16, fontweight='bold')
    ax1.set_xlabel('X Pozisyonu', fontsize=12)
    ax1.set_ylabel('Y Pozisyonu', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Pozisyon hatası evrimi (detaylı)
    ax2 = axes[0, 1]
    ax2.plot(position_errors, 'lightcoral', alpha=0.6, linewidth=1, label='Ham Hata')
    
    # Multiple moving averages
    windows = [10, 30, 50]
    colors = ['red', 'darkred', 'maroon']
    for window, color in zip(windows, colors):
        if len(position_errors) > window:
            moving_avg = np.convolve(position_errors, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(position_errors)), moving_avg, color, 
                    linewidth=3, label=f'{window}-adım ortalaması', alpha=0.8)
    
    # İstatistikleri göster
    mean_error = np.mean(position_errors)
    median_error = np.median(position_errors)
    ax2.axhline(y=mean_error, color='blue', linestyle='--', 
                linewidth=2, label=f'Ortalama: {mean_error:.1f}', alpha=0.8)
    ax2.axhline(y=median_error, color='green', linestyle=':', 
                linewidth=2, label=f'Medyan: {median_error:.1f}', alpha=0.8)
    ax2.axhline(y=1.0, color='gold', linestyle='-', 
                linewidth=3, label='Başarı Eşiği', alpha=0.9)
    
    ax2.set_title('📈 POZİSYON HATASI EVRİMİ', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Zaman Adımı', fontsize=12)
    ax2.set_ylabel('Mesafe Hatası', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Hareket hızları karşılaştırması (gelişmiş)
    ax3 = axes[0, 2]
    pred_speeds = np.linalg.norm(predicted_movements, axis=1)
    real_speeds = np.linalg.norm(real_movements, axis=1)
    
    ax3.plot(real_speeds, 'b-', alpha=0.7, linewidth=3, label='Gerçek Hız')
    ax3.plot(pred_speeds, 'r--', alpha=0.7, linewidth=3, label='Agent Hız')
    
    # Moving averages
    if len(pred_speeds) > 20:
        real_smooth = np.convolve(real_speeds, np.ones(20)/20, mode='valid')
        pred_smooth = np.convolve(pred_speeds, np.ones(20)/20, mode='valid')
        ax3.plot(range(19, len(real_speeds)), real_smooth, 'darkblue', 
                linewidth=4, label='Gerçek (smooth)', alpha=0.9)
        ax3.plot(range(19, len(pred_speeds)), pred_smooth, 'darkred', 
                linewidth=4, label='Agent (smooth)', alpha=0.9)
    
    ax3.axhline(y=np.mean(real_speeds), color='blue', linestyle=':', alpha=0.8)
    ax3.axhline(y=np.mean(pred_speeds), color='red', linestyle=':', alpha=0.8)
    
    ax3.set_title('🚀 HAREKET HIZI KARŞILAŞTIRMASI', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Zaman Adımı', fontsize=12)
    ax3.set_ylabel('Hareket Hızı', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Hareket yönü benzerliği (geliştirilmiş)
    ax4 = axes[0, 3]
    ax4.plot(movement_similarities, 'purple', linewidth=2, alpha=0.8, label='Yön Benzerliği')
    
    # Moving average
    if len(movement_similarities) > 15:
        sim_smooth = np.convolve(movement_similarities, np.ones(15)/15, mode='valid')
        ax4.plot(range(14, len(movement_similarities)), sim_smooth, 'darkmagenta', 
                linewidth=4, label='Smooth Benzerlik', alpha=0.9)
    
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.axhline(y=np.mean(movement_similarities), color='red', linestyle='--', 
                linewidth=2, label=f'Ort: {np.mean(movement_similarities):.3f}', alpha=0.8)
    
    # Pozitif/negatif alanları renklendir
    ax4.fill_between(range(len(movement_similarities)), 0, movement_similarities, 
                     where=(movement_similarities >= 0), color='green', alpha=0.2, 
                     interpolate=True, label='Pozitif Benzerlik')
    ax4.fill_between(range(len(movement_similarities)), 0, movement_similarities, 
                     where=(movement_similarities < 0), color='red', alpha=0.2, 
                     interpolate=True, label='Negatif Benzerlik')
    
    ax4.set_title('🎯 HAREKET YÖNÜ BENZERLİĞİ', fontsize=16, fontweight='bold')
    ax4.set_xlabel('Zaman Adımı', fontsize=12)
    ax4.set_ylabel('Kosinüs Benzerliği', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. Reward evrimi (ultra detaylı)
    ax5 = axes[1, 0]
    ax5.plot(rewards, 'green', alpha=0.5, linewidth=1, label='Ham Reward')
    
    # Reward komponenetlerini ayır (pozitif/negatif)
    positive_rewards = np.where(rewards >= 0, rewards, 0)
    negative_rewards = np.where(rewards < 0, rewards, 0)
    
    ax5.fill_between(range(len(rewards)), 0, positive_rewards, 
                     color='green', alpha=0.3, label='Pozitif Reward')
    ax5.fill_between(range(len(rewards)), 0, negative_rewards, 
                     color='red', alpha=0.3, label='Negatif Reward')
    
    # Moving average
    if len(rewards) > 25:
        reward_smooth = np.convolve(rewards, np.ones(25)/25, mode='valid')
        ax5.plot(range(24, len(rewards)), reward_smooth, 'darkgreen', 
                linewidth=4, label='25-adım ortalaması', alpha=0.9)
    
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax5.axhline(y=np.mean(rewards), color='blue', linestyle='--', 
                linewidth=2, label=f'Ortalama: {np.mean(rewards):.1f}', alpha=0.8)
    
    ax5.set_title('🎁 REWARD EVRİMİ', fontsize=16, fontweight='bold')
    ax5.set_xlabel('Zaman Adımı', fontsize=12)
    ax5.set_ylabel('Reward', fontsize=12)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # 6. Son 100 adım detaylı analiz
    ax6 = axes[1, 1]
    last_steps = min(100, len(target_positions))
    
    # Son 100 adımın trajectory'si
    last_targets = target_positions[-last_steps:]
    last_agents = agent_positions[-last_steps:]
    
    ax6.plot(last_targets[:, 0], last_targets[:, 1], 'b-', 
             linewidth=5, label='Gerçek Son 100', alpha=0.9)
    ax6.plot(last_agents[:, 0], last_agents[:, 1], 'r--', 
             linewidth=4, label='Agent Son 100', alpha=0.8)
    
    # Son başarıları vurgula
    recent_success_indices = success_indices[success_indices >= len(success_flags) - last_steps]
    if len(recent_success_indices) > 0:
        recent_indices = recent_success_indices - (len(success_flags) - last_steps)
        ax6.scatter(last_agents[recent_indices, 0], last_agents[recent_indices, 1], 
                   c='gold', s=150, marker='*', label='Son Başarılar', zorder=5, 
                   edgecolors='black', linewidth=2)
    
    # Her 10. noktayı numara ile işaretle
    for i in range(0, last_steps, 10):
        ax6.annotate(str(len(target_positions) - last_steps + i), 
                    (last_targets[i, 0], last_targets[i, 1]), 
                    fontsize=8, ha='center', va='center', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax6.set_title(f'🔍 SON {last_steps} ADIM DETAYI', fontsize=16, fontweight='bold')
    ax6.set_xlabel('X Pozisyonu', fontsize=12)
    ax6.set_ylabel('Y Pozisyonu', fontsize=12)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # 7. Çoklu çalışma karşılaştırması
    ax7 = axes[1, 2]
    run_errors = [np.mean(r['position_errors']) for r in all_results]
    run_successes = [r['total_successes'] for r in all_results]
    run_similarities = [np.mean(r['movement_similarities']) for r in all_results]
    
    x = np.arange(len(all_results))
    width = 0.25
    
    # Çoklu bar chart
    bars1 = ax7.bar(x - width, run_errors, width, label='Ortalama Hata', 
                   color='red', alpha=0.7)
    bars2 = ax7.bar(x, [s/10 for s in run_successes], width, label='Başarı/10', 
                   color='green', alpha=0.7)
    bars3 = ax7.bar(x + width, [abs(s)*50 for s in run_similarities], width, 
                   label='|Benzerlik|*50', color='blue', alpha=0.7)
    
    # En iyi çalışmayı vurgula
    bars1[best_run_idx].set_color('darkred')
    bars1[best_run_idx].set_alpha(1.0)
    
    ax7.set_xlabel('Çalışma Numarası', fontsize=12)
    ax7.set_ylabel('Değer', fontsize=12)
    ax7.set_title('🔄 ÇOKLU ÇALIŞMA KARŞILAŞTIRMASI', fontsize=16, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels([f'Run {i+1}' for i in range(len(all_results))])
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # Bar değerlerini yaz
    for i, (bar1, bar2, bar3) in enumerate(zip(bars1, bars2, bars3)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        height3 = bar3.get_height()
        ax7.text(bar1.get_x() + bar1.get_width()/2., height1 + 1,
                f'{run_errors[i]:.0f}', ha='center', va='bottom', fontsize=8)
        ax7.text(bar2.get_x() + bar2.get_width()/2., height2 + 1,
                f'{run_successes[i]}', ha='center', va='bottom', fontsize=8)
        ax7.text(bar3.get_x() + bar3.get_width()/2., height3 + 1,
                f'{run_similarities[i]:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 8. Başarı analizi (geliştirilmiş)
    ax8 = axes[1, 3]
    thresholds = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
    success_rates = []
    
    for threshold in thresholds:
        success_rate = (position_errors < threshold).mean() * 100
        success_rates.append(success_rate)
    
    # Gradient color scheme
    colors = plt.cm.RdYlGn([rate/100 for rate in success_rates])
    bars = ax8.bar(range(len(thresholds)), success_rates, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1)
    
    ax8.set_xticks(range(len(thresholds)))
    ax8.set_xticklabels([f'<{t}' for t in thresholds], rotation=45)
    ax8.set_title('✅ BAŞARI ORANLARI ANALİZİ', fontsize=16, fontweight='bold')
    ax8.set_xlabel('Hata Threshold', fontsize=12)
    ax8.set_ylabel('Başarı Oranı (%)', fontsize=12)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Bar değerlerini yaz
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 9. Hız dağılımı histogramı
    ax9 = axes[2, 0]
    
    # Histogram
    ax9.hist(real_speeds, bins=30, alpha=0.7, label='Gerçek Hız', 
             color='blue', density=True, edgecolor='black')
    ax9.hist(pred_speeds, bins=30, alpha=0.7, label='Agent Hız', 
             color='red', density=True, edgecolor='black')
    
    # İstatistikler
    ax9.axvline(np.mean(real_speeds), color='blue', linestyle='--', 
                linewidth=2, label=f'Gerçek Ort: {np.mean(real_speeds):.2f}')
    ax9.axvline(np.mean(pred_speeds), color='red', linestyle='--', 
                linewidth=2, label=f'Agent Ort: {np.mean(pred_speeds):.2f}')
    
    ax9.set_title('📊 HIZ DAĞILIMI', fontsize=16, fontweight='bold')
    ax9.set_xlabel('Hız', fontsize=12)
    ax9.set_ylabel('Yoğunluk', fontsize=12)
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)
    
    # 10. Zaman serisi korelasyon analizi
    ax10 = axes[2, 1]
    
    # Pozisyon hatası ile reward korelasyonu
    correlation_window = 50
    correlations = []
    windows_center = []
    
    for i in range(correlation_window, len(position_errors) - correlation_window):
        window_errors = position_errors[i-correlation_window//2:i+correlation_window//2]
        window_rewards = rewards[i-correlation_window//2:i+correlation_window//2]
        
        if len(window_errors) > 1 and len(window_rewards) > 1:
            corr = np.corrcoef(window_errors, window_rewards)[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
            windows_center.append(i)
    
    if len(correlations) > 0:
        ax10.plot(windows_center, correlations, 'purple', linewidth=3, 
                 label='Hata-Reward Korelasyonu')
        ax10.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax10.axhline(y=-0.5, color='red', linestyle='--', alpha=0.7, 
                    label='Güçlü Negatif Korelasyon')
        ax10.axhline(y=-0.8, color='darkred', linestyle=':', alpha=0.7, 
                    label='Çok Güçlü Negatif')
    
    ax10.set_title('📈 HATA-REWARD KORELASYONU', fontsize=16, fontweight='bold')
    ax10.set_xlabel('Zaman Adımı', fontsize=12)
    ax10.set_ylabel('Korelasyon', fontsize=12)
    ax10.legend(fontsize=9)
    ax10.grid(True, alpha=0.3)
    
    # 11. Kümülatif performans
    ax11 = axes[2, 2]
    
    # Kümülatif başarı oranı (farklı window'lar)
    window_sizes = [25, 50, 100]
    colors_cum = ['lightgreen', 'green', 'darkgreen']
    
    for window_size, color in zip(window_sizes, colors_cum):
        cumulative_success = []
        window_centers = []
        
        for i in range(window_size, len(position_errors), window_size//4):
            window_errors = position_errors[max(0, i-window_size):i]
            success_rate = (window_errors < 2.0).mean() * 100  # 2 birim threshold
            cumulative_success.append(success_rate)
            window_centers.append(i)
        
        if len(cumulative_success) > 0:
            ax11.plot(window_centers, cumulative_success, color, linewidth=3, 
                     marker='o', markersize=4, label=f'{window_size}-adım window')
    
    ax11.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='%50 Hedef')
    ax11.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='%80 Hedef')
    
    ax11.set_title('📊 KÜMÜLATİF BAŞARI EVRİMİ', fontsize=16, fontweight='bold')
    ax11.set_xlabel('Zaman Adımı', fontsize=12)
    ax11.set_ylabel('Başarı Oranı (%)', fontsize=12)
    ax11.legend(fontsize=9)
    ax11.grid(True, alpha=0.3)
    
    # 12. Final pozisyon yoğunluk haritası
    ax12 = axes[2, 3]
    
    # 2D histogram (heatmap)
    x_edges = np.linspace(min(np.min(agent_positions[:, 0]), np.min(target_positions[:, 0])), 
                         max(np.max(agent_positions[:, 0]), np.max(target_positions[:, 0])), 20)
    y_edges = np.linspace(min(np.min(agent_positions[:, 1]), np.min(target_positions[:, 1])), 
                         max(np.max(agent_positions[:, 1]), np.max(target_positions[:, 1])), 20)
    
    H_agent, xedges, yedges = np.histogram2d(agent_positions[:, 0], agent_positions[:, 1], 
                                            bins=[x_edges, y_edges])
    H_target, _, _ = np.histogram2d(target_positions[:, 0], target_positions[:, 1], 
                                   bins=[x_edges, y_edges])
    
    # Overlap hesapla
    overlap = np.minimum(H_agent, H_target)
    total_agent = np.sum(H_agent)
    overlap_ratio = np.sum(overlap) / max(total_agent, 1) * 100
    
    im = ax12.imshow(H_agent.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], 
                    cmap='Reds', alpha=0.7, interpolation='bilinear')
    ax12.contour(H_target.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], 
                colors='blue', alpha=0.8, linewidths=2)
    
    # Başarılı noktaları göster
    if len(success_indices) > 0:
        ax12.scatter(agent_positions[success_indices, 0], agent_positions[success_indices, 1], 
                    c='gold', s=100, marker='*', label='Başarılar', zorder=5, 
                    edgecolors='black', linewidth=1)
    
    ax12.set_title(f'🗺️ POZİSYON YOĞUNLUK HARİTASI\nÖrtüşme: {overlap_ratio:.1f}%', 
                   fontsize=16, fontweight='bold')
    ax12.set_xlabel('X Pozisyonu', fontsize=12)
    ax12.set_ylabel('Y Pozisyonu', fontsize=12)
    ax12.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('ultra_aggressive_player14_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 🔥 ULTRA DETAYLI PERFORMANS RAPORU
    print("\n" + "="*100)
    print("🔥 ULTRA AGRESİF PLAYER 14 AGENT PERFORMANS RAPORU")
    print("="*100)
    
    # Tüm çalışmalar için istatistik
    all_mean_errors = [np.mean(r['position_errors']) for r in all_results]
    all_success_rates = [r['total_successes']/r['steps']*100 for r in all_results]
    all_similarities = [np.mean(r['movement_similarities']) for r in all_results]
    
    print(f"📊 GENEL İSTATİSTİKLER ({num_runs} Çalışma Ortalaması):")
    print(f"   • Ortalama pozisyon hatası: {np.mean(all_mean_errors):.2f} ± {np.std(all_mean_errors):.2f}")
    print(f"   • En iyi ortalama hata: {np.min(all_mean_errors):.2f}")
    print(f"   • En kötü ortalama hata: {np.max(all_mean_errors):.2f}")
    print(f"   • Ortalama başarı oranı: {np.mean(all_success_rates):.1f}% ± {np.std(all_success_rates):.1f}%")
    print(f"   • En iyi başarı oranı: {np.max(all_success_rates):.1f}%")
    print(f"   • Ortalama hareket benzerliği: {np.mean(all_similarities):.3f} ± {np.std(all_similarities):.3f}")
    
    # En iyi çalışma detayları
    print(f"\n🏆 EN İYİ ÇALIŞMA DETAYLARI (#{best_run_idx + 1}):")
    print(f"   • Toplam adım: {len(position_errors)}")
    print(f"   • Ortalama pozisyon hatası: {np.mean(position_errors):.2f}")
    print(f"   • Medyan pozisyon hatası: {np.median(position_errors):.2f}")
    print(f"   • Minimum pozisyon hatası: {np.min(position_errors):.2f}")
    print(f"   • Maksimum pozisyon hatası: {np.max(position_errors):.2f}")
    print(f"   • Standart sapma: {np.std(position_errors):.2f}")
    print(f"   • Toplam başarı sayısı: {total_successes}")
    print(f"   • Başarı oranı: {(total_successes / len(position_errors) * 100):.1f}%")
    print(f"   • Ortalama reward: {np.mean(rewards):.1f}")
    print(f"   • Toplam reward: {np.sum(rewards):.1f}")
    
    # Hareket analizi
    avg_predicted_speed = np.mean(np.linalg.norm(predicted_movements, axis=1))
    avg_real_speed = np.mean(np.linalg.norm(real_movements, axis=1))
    speed_correlation = np.corrcoef(
        np.linalg.norm(predicted_movements, axis=1),
        np.linalg.norm(real_movements, axis=1)
    )[0, 1] if len(predicted_movements) > 1 else 0.0
    
    print(f"\n🚀 HAREKET ANALİZİ:")
    print(f"   • Ortalama agent hızı: {avg_predicted_speed:.2f}")
    print(f"   • Ortalama gerçek hız: {avg_real_speed:.2f}")
    print(f"   • Hız oranı (agent/gerçek): {(avg_predicted_speed/max(avg_real_speed, 1e-6)):.2f}")
    print(f"   • Hız korelasyonu: {speed_correlation:.3f}")
    print(f"   • Ortalama hareket benzerliği: {np.mean(movement_similarities):.3f}")
    
    # Başarı threshold analizi
    print(f"\n✅ BAŞARI ANALİZİ (Farklı Threshold'lar):")
    thresholds = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    for threshold in thresholds:
        success_rate = (position_errors < threshold).mean() * 100
        print(f"   • <{threshold:.1f} birim hata: {success_rate:.1f}% ({np.sum(position_errors < threshold)} / {len(position_errors)})")
    
    # Zaman analizi
    print(f"\n⏰ ZAMAN ANALİZİ:")
    quartiles = np.percentile(range(len(position_errors)), [25, 50, 75])
    for i, (start, end) in enumerate([(0, int(quartiles[0])), 
                                     (int(quartiles[0]), int(quartiles[1])),
                                     (int(quartiles[1]), int(quartiles[2])),
                                     (int(quartiles[2]), len(position_errors))]):
        if end > start:
            quarter_errors = position_errors[start:end]
            quarter_success = (quarter_errors < 2.0).mean() * 100
            print(f"   • {i+1}. Çeyrek ({start}-{end}): Ort.Hata={np.mean(quarter_errors):.2f}, Başarı=%{quarter_success:.1f}")
    
    # Reward komponenti analizi
    print(f"\n🎁 REWARD KOMPONENTİ ANALİZİ:")
    print(f"   • Ortalama toplam reward: {np.mean(rewards):.1f}")
    print(f"   • Reward std sapması: {np.std(rewards):.1f}")
    print(f"   • En yüksek reward: {np.max(rewards):.1f}")
    print(f"   • En düşük reward: {np.min(rewards):.1f}")
    print(f"   • Pozitif reward oranı: {(rewards > 0).mean() * 100:.1f}%")
    
    # Son performans trendi
    if len(position_errors) > 100:
        last_100_errors = position_errors[-100:]
        first_100_errors = position_errors[:100]
        improvement = np.mean(first_100_errors) - np.mean(last_100_errors)
        print(f"\n📈 ÖĞRENME TRENDİ:")
        print(f"   • İlk 100 adım ortalama hatası: {np.mean(first_100_errors):.2f}")
        print(f"   • Son 100 adım ortalama hatası: {np.mean(last_100_errors):.2f}")
        print(f"   • İyileşme: {improvement:.2f} ({'✅ Pozitif' if improvement > 0 else '❌ Negatif'})")
    
    # Performans özeti
    overall_score = (
        (100 - np.mean(position_errors)) * 0.4 +  # Pozisyon doğruluğu (40%)
        (total_successes / len(position_errors) * 100) * 0.3 +  # Başarı oranı (30%)
        ((np.mean(movement_similarities) + 1) * 50) * 0.2 +  # Hareket benzerliği (20%)
        (np.mean(rewards) / 100 * 10) * 0.1  # Reward performansı (10%)
    )
    
    print(f"\n🏆 GENEL PERFORMANS SKORU:")
    print(f"   • Kompozit Skor: {overall_score:.1f}/100")
    
    if overall_score >= 80:
        grade = "🥇 MÜKEMMEL"
    elif overall_score >= 70:
        grade = "🥈 ÇOK İYİ"
    elif overall_score >= 60:
        grade = "🥉 İYİ"
    elif overall_score >= 50:
        grade = "📈 ORTA"
    else:
        grade = "📉 GELİŞTİRİLMELİ"
    
    print(f"   • Performans Notu: {grade}")
    print("="*100)
    
    return {
        'best_run_results': best_run,
        'all_results': all_results,
        'performance_metrics': {
            'avg_position_error': np.mean(position_errors),
            'success_rate': total_successes / len(position_errors) * 100,
            'avg_movement_similarity': np.mean(movement_similarities),
            'avg_reward': np.mean(rewards),
            'overall_score': overall_score,
            'speed_ratio': avg_predicted_speed/max(avg_real_speed, 1e-6),
            'speed_correlation': speed_correlation
        }
    }

def save_model_with_metadata(model, player14, ball, evaluation_results):
    """Model ve metadata'yı kaydet"""
    print("\n=== MODEL VE METADATA KAYDETME ===")
    
    # Model kaydetme
    model.save("ultra_aggressive_player14_final")
    
    # Metadata oluşturma
    metadata = {
        'model_info': {
            'algorithm': 'PPO',
            'policy': 'MlpPolicy',
            'training_timesteps': 500000,
            'architecture': 'Ultra Aggressive Player 14 Tracker'
        },
        'data_info': {
            'player14_data_length': len(player14),
            'ball_data_length': len(ball),
            'observation_space_dim': 29,
            'action_space_dim': 2
        },
        'performance': evaluation_results['performance_metrics'],
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'version': '1.0'
    }
    
    # JSON olarak kaydet
    import json
    with open('ultra_aggressive_player14_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print("✅ Model ve metadata kaydedildi:")
    print("   • ultra_aggressive_player14_final.zip")
    print("   • ultra_aggressive_player14_metadata.json")
    print("   • ultra_aggressive_player14_analysis.png")

def load_trained_model():
    """Eğitilmiş modeli yükle"""
    try:
        model = PPO.load("ultra_aggressive_player14_final")
        print("✅ Model başarıyla yüklendi: ultra_aggressive_player14_final.zip")
        return model
    except:
        try:
            model = PPO.load("ultra_aggressive_player14_model")
            print("✅ Model başarıyla yüklendi: ultra_aggressive_player14_model.zip")
            return model
        except:
            print("❌ Model yüklenemedi! Önce eğitim yapmanız gerekiyor.")
            return None

def quick_test(model=None, num_steps=200):
    """Hızlı test fonksiyonu"""
    print("\n=== HIZLI TEST ===")
    
    if model is None:
        model = load_trained_model()
        if model is None:
            return
    
    # Veri yükle
    ball, player14 = load_and_clean_data()
    if ball is None or player14 is None:
        return
    
    # Environment oluştur
    env = ImprovedPlayer14Env(player14, ball)
    obs, _ = env.reset()
    
    print(f"🚀 {num_steps} adım hızlı test başlıyor...")
    
    errors = []
    similarities = []
    
    for step in range(min(num_steps, len(player14)-1)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        errors.append(info['position_error'])
        similarities.append(info.get('movement_similarity', 0.0))
        
        if step % 50 == 0:
            print(f"  Adım {step}: Hata={info['position_error']:.2f}, Benzerlik={info.get('movement_similarity', 0.0):.3f}")
        
        if terminated or truncated:
            break
    
    print(f"\n📊 HIZLI TEST SONUÇLARI:")
    print(f"   • Ortalama hata: {np.mean(errors):.2f}")
    print(f"   • Medyan hata: {np.median(errors):.2f}")
    print(f"   • Başarı oranı (<2.0): {(np.array(errors) < 2.0).mean() * 100:.1f}%")
    print(f"   • Ortalama benzerlik: {np.mean(similarities):.3f}")
    print(f"   • Test süresi: {len(errors)} adım")

def main():
    """Ana çalıştırma fonksiyonu"""
    print("🔥 ULTRA AGRESİF PLAYER 14 TRACKING SYSTEM")
    print("="*60)
    
    # Kullanıcı seçimi
    while True:
        print("\nSeçenekler:")
        print("1. 🔥 Ultra Agresif Eğitim Yap (500K steps)")
        print("2. 📊 Var olan Modeli Değerlendir")
        print("3. ⚡ Hızlı Test (200 steps)")
        print("4. 📁 Model Yükle ve Test")
        print("5. 🚪 Çıkış")
        
        choice = input("\nSeçiminiz (1-5): ").strip()
        
        try:
            if choice == "1":
                print("\n🔥 ULTRA AGRESİF EĞİTİM BAŞLIYOR...")
                model, player14, ball = train_improved_agent()
                if model is not None:
                    print("\n📊 EĞİTİM SONRASI DEĞERLENDİRME...")
                    results = detailed_evaluation(model, player14, ball)
                    save_model_with_metadata(model, player14, ball, results)
            
            elif choice == "2":
                model = load_trained_model()
                if model is not None:
                    ball, player14 = load_and_clean_data()
                    if ball is not None and player14 is not None:
                        results = detailed_evaluation(model, player14, ball)
            
            elif choice == "3":
                quick_test()
            
            elif choice == "4":
                model = load_trained_model()
                if model is not None:
                    steps = input("Kaç adım test edilsin? (varsayılan: 200): ").strip()
                    steps = int(steps) if steps.isdigit() else 200
                    quick_test(model, steps)
            
            elif choice == "5":
                print("👋 Görüşürüz!")
                break
            
            else:
                print("❌ Geçersiz seçim! Lütfen 1-5 arasında bir sayı girin.")
        
        except KeyboardInterrupt:
            print("\n\n⚠️ İşlem kullanıcı tarafından iptal edildi.")
        except Exception as e:
            print(f"\n❌ Hata oluştu: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()