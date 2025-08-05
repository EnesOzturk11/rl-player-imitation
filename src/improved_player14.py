import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt

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
        
        # ✅ FİX 1: DAHA GENİŞ VE REALİSTİK ACTION SPACE
        pos = self.player14_data[["position_x", "position_y"]].values
        if len(pos) > 1:
            dx = np.diff(pos[:, 0])
            dy = np.diff(pos[:, 1])
            speeds = np.sqrt(dx**2 + dy**2)
            
            # Gerçek hareket istatistikleri
            max_speed = np.percentile(speeds, 98)  # %98'lik dilim (daha geniş)
            mean_speed = np.mean(speeds)
            std_speed = np.std(speeds)
            
            print(f"Player 14 max hız (98th percentile): {max_speed:.2f}")
            print(f"Player 14 ortalama hız: {mean_speed:.2f}")
            print(f"Player 14 hız std: {std_speed:.2f}")
            
            # Daha geniş action space - gerçek hareketlerin 2 katına kadar
            speed_limit = max(max_speed * 2.0, mean_speed + 4 * std_speed)
        else:
            speed_limit = 10.0
        
        print(f"Action space hız limiti: {speed_limit:.2f}")
        
        # Geniş action space
        self.action_space = spaces.Box(
            low=np.array([-speed_limit, -speed_limit], dtype=np.float32), 
            high=np.array([speed_limit, speed_limit], dtype=np.float32),
            dtype=np.float32
        )
        
        # ✅ FİX 2: ZENGİNLEŞTİRİLMİŞ OBSERVATION SPACE
        # [agent_x, agent_y, ball_x, ball_y, ball_direction, 
        #  relative_ball_x, relative_ball_y, prev_dx, prev_dy, 
        #  target_next_x, target_next_y, distance_to_target,
        #  target_direction, speed_to_target, time_remaining]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
        
        self.current_step = 0
        self.agent_pos = None
        self.prev_movement = np.array([0.0, 0.0])
        self.total_distance_traveled = 0.0
        
        # Normalizasyon için scale faktörleri
        self.pos_scale = 1.0 / (np.std(pos.flatten()) + 1e-8) if len(pos) > 0 else 1.0
        self.max_distance = np.linalg.norm([np.ptp(pos[:, 0]), np.ptp(pos[:, 1])]) if len(pos) > 0 else 100.0
        
        print(f"Pozisyon scale faktörü: {self.pos_scale:.4f}")
        print(f"Maksimum mesafe: {self.max_distance:.2f}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.total_distance_traveled = 0.0
        
        # Player 14'ün başlangıç pozisyonu
        initial_pos = self.player14_data[["position_x", "position_y"]].iloc[0].values
        self.agent_pos = initial_pos.copy()
        self.prev_movement = np.array([0.0, 0.0])
        
        # NaN kontrolü
        self.agent_pos = np.nan_to_num(self.agent_pos, nan=0.0)
        
        # Zengin observation
        observation = self._get_observation()
        
        info = {'step': self.current_step}
        return observation, info
    
    def _get_observation(self):
        """✅ FİX 3: ZENGİNLEŞTİRİLMİŞ GÖZLEM ALANI"""
        # Mevcut ball pozisyonu
        if self.current_step < len(self.ball_data):
            ball_pos = self.ball_data[["position_x", "position_y"]].iloc[self.current_step].values
            ball_dir = np.array([self.ball_data["direction_deg"].iloc[self.current_step]])
        else:
            ball_pos = self.ball_data[["position_x", "position_y"]].iloc[-1].values
            ball_dir = np.array([self.ball_data["direction_deg"].iloc[-1]])
        
        # NaN kontrolü
        ball_pos = np.nan_to_num(ball_pos, nan=0.0)
        ball_dir = np.nan_to_num(ball_dir, nan=0.0)
        
        # Göreceli ball pozisyonu
        relative_ball = ball_pos - self.agent_pos
        
        # Bir sonraki target pozisyonu
        if self.current_step + 1 < len(self.player14_data):
            next_target = self.player14_data[["position_x", "position_y"]].iloc[self.current_step + 1].values
        else:
            next_target = self.player14_data[["position_x", "position_y"]].iloc[-1].values
        
        next_target = np.nan_to_num(next_target, nan=0.0)
        
        # Ek özellikler
        distance_to_target = np.linalg.norm(next_target - self.agent_pos)
        target_direction = np.arctan2(next_target[1] - self.agent_pos[1], 
                                    next_target[0] - self.agent_pos[0])
        
        # Hedefe ulaşmak için gereken hız
        speed_to_target = distance_to_target  # Bir adımda ulaşmak için gereken hız
        
        # Kalan zaman (normalize edilmiş)
        time_remaining = (self.max_steps - self.current_step) / self.max_steps
        
        # Zengin observation vektörü
        observation = np.concatenate([
            self.agent_pos * self.pos_scale,           # Normalize agent pozisyonu
            ball_pos * self.pos_scale,                 # Normalize ball pozisyonu
            ball_dir / 360.0,                          # Normalize ball yönü
            relative_ball * self.pos_scale,            # Göreceli ball pozisyonu
            self.prev_movement,                        # Önceki hareket
            next_target * self.pos_scale,              # Hedef pozisyon
            [distance_to_target / self.max_distance],  # Hedefe mesafe (normalize)
            [target_direction / np.pi],                # Hedef yönü (normalize)
            [speed_to_target / self.max_distance],     # Gereken hız (normalize)
            [time_remaining]                           # Kalan zaman
        ])
        
        return np.nan_to_num(observation, nan=0.0).astype(np.float32)

    def step(self, action):
        # Action'ı temizle ve sınırla
        action = np.nan_to_num(action, nan=0.0)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # ✅ FİX 4: HAREKET ETMEYİ TEŞVIK ET
        # Eğer hareket çok küçükse, biraz büyüt
        action_magnitude = np.linalg.norm(action)
        if action_magnitude < 0.1:  # Çok küçük hareketleri büyüt
            if action_magnitude > 0:
                action = action / action_magnitude * 0.5  # Minimum hareket
        
        # Agent'ı hareket ettir
        old_pos = self.agent_pos.copy()
        self.agent_pos += action
        self.prev_movement = action.copy()
        self.total_distance_traveled += np.linalg.norm(action)
        self.current_step += 1
        
        # Player 14'ün bu adımdaki gerçek pozisyonu
        if self.current_step < len(self.player14_data):
            target_pos = self.player14_data[["position_x", "position_y"]].iloc[self.current_step].values
            prev_target_pos = self.player14_data[["position_x", "position_y"]].iloc[self.current_step-1].values
            real_movement = target_pos - prev_target_pos
        else:
            target_pos = self.player14_data[["position_x", "position_y"]].iloc[-1].values
            if len(self.player14_data) > 1:
                prev_pos = self.player14_data[["position_x", "position_y"]].iloc[-2].values
                real_movement = target_pos - prev_pos
            else:
                real_movement = np.array([0.0, 0.0])
        
        # NaN kontrolü
        target_pos = np.nan_to_num(target_pos, nan=0.0)
        self.agent_pos = np.nan_to_num(self.agent_pos, nan=0.0)
        real_movement = np.nan_to_num(real_movement, nan=0.0)
        
        # ✅ FİX 5: GELİŞTİRİLMİŞ REWARD SİSTEMİ
        
        # 1. Pozisyon hatası (ağırlıklı - daha önemli)
        position_error = np.linalg.norm(self.agent_pos - target_pos)
        position_reward = -position_error * 2.0  # Daha yüksek ağırlık
        
        # 2. Hareket yönü benzerliği (teşvik et)
        movement_similarity = np.dot(action, real_movement) / (np.linalg.norm(action) * np.linalg.norm(real_movement) + 1e-8)
        movement_reward = movement_similarity * 1.5  # Pozitif reward
        
        # 3. Hareket etme teşviki (durağanlığı önle)
        movement_magnitude = np.linalg.norm(action)
        movement_bonus = min(movement_magnitude * 0.1, 1.0)  # Hareket et!
        
        # 4. Yakınlık bonusu (hedefe yaklaştıkça artan bonus)
        max_possible_distance = self.max_distance
        proximity_bonus = max(0, (max_possible_distance - position_error) / max_possible_distance) * 3.0
        
        # 5. İlerleme bonusu (hedefe yaklaşıyorsa bonus)
        if hasattr(self, 'prev_distance_to_target'):
            distance_improvement = self.prev_distance_to_target - position_error
            progress_bonus = distance_improvement * 2.0  # İlerleme için büyük bonus
        else:
            progress_bonus = 0.0
        
        self.prev_distance_to_target = position_error
        
        # 6. Zaman bonusu (erken bitirme için)
        time_bonus = 0.05  # Her adım için küçük bonus
        
        # 7. Hedefe çok yaklaşma bonusu
        if position_error < 2.0:
            close_bonus = (2.0 - position_error) * 5.0  # Çok yaklaşırsa büyük bonus
        else:
            close_bonus = 0.0
        
        # 8. Son adımlarda daha büyük ödül
        final_steps_multiplier = 1.0
        if self.current_step > self.max_steps * 0.8:  # Son %20'de
            final_steps_multiplier = 2.0
        
        # Toplam reward
        total_reward = (
            position_reward * final_steps_multiplier + 
            movement_reward + 
            movement_bonus + 
            proximity_bonus * final_steps_multiplier + 
            progress_bonus * final_steps_multiplier + 
            time_bonus + 
            close_bonus * final_steps_multiplier
        )
        
        # ✅ FİX 6: BAŞARI DURUMUNDA ERKEN BİTİRME
        success_threshold = 1.0  # 1 birimden yakınsa başarılı
        terminated = False
        
        if position_error < success_threshold:
            total_reward += 50.0  # Büyük başarı bonusu
            terminated = True
            print(f"🎯 Başarı! Adım {self.current_step}'te hedefe ulaşıldı (hata: {position_error:.2f})")
        elif self.current_step >= self.max_steps:
            terminated = True
            # Son pozisyona ne kadar yakın kaldıysa o kadar bonus
            final_bonus = max(0, 20 - position_error)
            total_reward += final_bonus
        
        truncated = False
        
        # Yeni gözlem
        observation = self._get_observation()
        
        info = {
            'step': self.current_step,
            'position_error': position_error,
            'movement_similarity': movement_similarity,
            'target_pos': target_pos,
            'agent_pos': self.agent_pos.copy(),
            'real_movement': real_movement,
            'predicted_movement': action,
            'position_reward': position_reward,
            'movement_reward': movement_reward,
            'proximity_bonus': proximity_bonus,
            'progress_bonus': progress_bonus,
            'total_distance_traveled': self.total_distance_traveled,
            'success': position_error < success_threshold
        }
        
        return observation, total_reward, terminated, truncated, info

def load_and_clean_data():
    """Player 14 ve ball verisini yükle ve temizle"""
    print("=== VERİ YÜKLEME ===")
    
    ball = pd.read_csv("data/ball_clean.csv")
    player14 = pd.read_csv("data/tracker_14_clean.csv")
    
    print(f"Ball veri boyutu: {ball.shape}")
    print(f"Player 14 veri boyutu: {player14.shape}")
    
    # Veri temizleme
    def clean_dataframe(df):
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
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
    """✅ FİX 7: GELİŞTİRİLMİŞ EĞİTİM PARAMETRELERİ"""
    print("=== GELİŞTİRİLMİŞ PLAYER 14 AGENT EĞİTİMİ ===")
    
    # Veri yükleme
    ball, player14 = load_and_clean_data()
    
    # Environment oluşturma
    env = DummyVecEnv([make_env(player14, ball)])
    
    # Daha agresif öğrenme parametreleri
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,      # Daha yüksek learning rate
        n_steps=1024,            # Daha küçük batch (daha sık güncelleme)
        batch_size=64,           # Daha küçük batch size
        n_epochs=20,             # Daha fazla epoch
        gamma=0.95,              # Daha düşük discount (yakın vadeli düşünmek için)
        gae_lambda=0.90,
        clip_range=0.3,          # Daha geniş clip range
        ent_coef=0.05,           # Daha fazla exploration
        vf_coef=0.5,             
        max_grad_norm=1.0,       
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])]  # Daha büyük network
        ),
        tensorboard_log="./tensorboard_logs_fixed/"
    )
    
    # Evaluation callback
    eval_env = DummyVecEnv([make_env(player14, ball)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models_fixed/",
        log_path="./logs_fixed/",
        eval_freq=3000,
        deterministic=True,
        render=False
    )
    
    # Daha uzun eğitim
    print("🚀 Geliştirilmiş eğitim başlıyor...")
    model.learn(
        total_timesteps=200000,  # Çok daha uzun eğitim
        callback=eval_callback,
        progress_bar=True
    )
    
    # Model kaydetme
    model.save("fixed_player14_model")
    print("✅ Düzeltilmiş model kaydedildi: fixed_player14_model.zip")
    
    return model, player14, ball

def detailed_evaluation(model, player14, ball):
    """Detaylı değerlendirme ve görselleştirme"""
    print("\n=== DETAYLI DEĞERLENDİRME ===")
    
    # Test environment
    env = ImprovedPlayer14Env(player14, ball)
    
    # Tahmin yapma
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
    test_length = min(len(player14) - 1, 500)
    print(f"Test uzunluğu: {test_length} adım")
    
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
        movement_similarities.append(info['movement_similarity'])
        predicted_movements.append(info['predicted_movement'].copy())
        real_movements.append(info['real_movement'].copy())
        success_flags.append(info['success'])
        
        if info['success']:
            total_successes += 1
        
        if terminated or truncated:
            print(f"Episode {step+1} adımda bitti")
            break
    
    # NumPy arrays'e çevir
    agent_positions = np.array(agent_positions)
    target_positions = np.array(target_positions)
    position_errors = np.array(position_errors)
    movement_similarities = np.array(movement_similarities)
    predicted_movements = np.array(predicted_movements)
    real_movements = np.array(real_movements)
    success_flags = np.array(success_flags)
    
    # ✅ FİX 8: GELİŞTİRİLMİŞ GÖRSELLEŞTİRME
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    fig.suptitle('🎯 DÜZELTILMIŞ Player 14 Agent Performans Analizi', fontsize=20, fontweight='bold')
    
    # 1. Tam yörünge karşılaştırması
    ax1 = axes[0, 0]
    ax1.plot(target_positions[:, 0], target_positions[:, 1], 'b-', 
             linewidth=4, label='Gerçek Player 14', alpha=0.9)
    ax1.plot(agent_positions[:, 0], agent_positions[:, 1], 'r--', 
             linewidth=3, label='AI Agent', alpha=0.8)
    
    # Başarılı noktaları vurgula
    success_indices = np.where(success_flags)[0]
    if len(success_indices) > 0:
        ax1.scatter(agent_positions[success_indices, 0], agent_positions[success_indices, 1], 
                   c='gold', s=100, marker='*', label=f'Başarı ({len(success_indices)})', zorder=5)
    
    ax1.scatter(target_positions[0, 0], target_positions[0, 1], 
                c='green', s=150, marker='o', label='Başlangıç', zorder=5)
    ax1.scatter(target_positions[-1, 0], target_positions[-1, 1], 
                c='red', s=150, marker='x', label='Bitiş', zorder=5)
    ax1.set_title('Tam Yörünge Karşılaştırması', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Pozisyonu')
    ax1.set_ylabel('Y Pozisyonu')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Pozisyon hatası evrimi
    ax2 = axes[0, 1]
    ax2.plot(position_errors, 'lightcoral', alpha=0.6, linewidth=1, label='Ham hata')
    
    # Moving average
    window = 30
    if len(position_errors) > window:
        moving_avg = np.convolve(position_errors, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(position_errors)), moving_avg, 'red', 
                linewidth=3, label=f'{window}-adım ortalaması')
    
    ax2.axhline(y=np.mean(position_errors), color='blue', linestyle='--', 
                linewidth=2, label=f'Genel ortalama: {np.mean(position_errors):.1f}')
    
    # Başarı threshold'u göster
    ax2.axhline(y=1.0, color='green', linestyle=':', 
                linewidth=2, label='Başarı Eşiği (1.0)')
    
    ax2.set_title('Pozisyon Hatası Evrimi', fontsize=14)
    ax2.set_xlabel('Zaman Adımı')
    ax2.set_ylabel('Mesafe Hatası')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Hareket hızları
    ax3 = axes[0, 2]
    pred_speeds = np.linalg.norm(predicted_movements, axis=1)
    real_speeds = np.linalg.norm(real_movements, axis=1)
    
    ax3.plot(real_speeds, 'b-', alpha=0.7, linewidth=2, label='Gerçek Hız')
    ax3.plot(pred_speeds, 'r--', alpha=0.7, linewidth=2, label='AI Hız')
    
    ax3.set_title('Hareket Hızı Karşılaştırması', fontsize=14)
    ax3.set_xlabel('Zaman Adımı')
    ax3.set_ylabel('Hareket Hızı')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Hareket benzerliği
    ax4 = axes[1, 0]
    ax4.plot(movement_similarities, 'purple', linewidth=2, alpha=0.8)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.axhline(y=np.mean(movement_similarities), color='red', linestyle='--', 
                linewidth=2, label=f'Ortalama: {np.mean(movement_similarities):.2f}')
    ax4.set_title('Hareket Yönü Benzerliği', fontsize=14)
    ax4.set_xlabel('Zaman Adımı')
    ax4.set_ylabel('Kosinüs Benzerliği')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Reward evrimi
    ax5 = axes[1, 1]
    ax5.plot(rewards, 'green', alpha=0.7, linewidth=1, label='Ham Reward')
    
    if len(rewards) > window:
        reward_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax5.plot(range(window-1, len(rewards)), reward_smooth, 'darkgreen', 
                linewidth=3, label=f'{window}-adım ortalaması')
    
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax5.axhline(y=np.mean(rewards), color='red', linestyle='--', 
                linewidth=2, label=f'Ortalama: {np.mean(rewards):.1f}')
    
    ax5.set_title('Reward Evrimi', fontsize=14)
    ax5.set_xlabel('Zaman Adımı')
    ax5.set_ylabel('Reward')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Son 100 adımın detaylı görünümü
    ax6 = axes[1, 2]
    last_steps = min(100, len(target_positions))
    ax6.plot(target_positions[-last_steps:, 0], target_positions[-last_steps:, 1], 
             'b-', linewidth=4, label='Gerçek Player 14', alpha=0.9)
    ax6.plot(agent_positions[-last_steps:, 0], agent_positions[-last_steps:, 1], 
             'r--', linewidth=3, label='AI Agent', alpha=0.8)
    
    # Son başarıları vurgula
    recent_successes = success_indices[success_indices >= len(success_flags) - last_steps] - (len(success_flags) - last_steps)
    if len(recent_successes) > 0:
        ax6.scatter(agent_positions[-last_steps:][recent_successes, 0], 
                   agent_positions[-last_steps:][recent_successes, 1], 
                   c='gold', s=100, marker='*', label='Son Başarılar', zorder=5)
    
    ax6.set_title(f'Son {last_steps} Adım Detayı', fontsize=14)
    ax6.set_xlabel('X Pozisyonu')
    ax6.set_ylabel('Y Pozisyonu')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Başarı analizi
    ax7 = axes[2, 0]
    thresholds = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    success_rates = []
    
    for threshold in thresholds:
        success_rate = (position_errors < threshold).mean() * 100
        success_rates.append(success_rate)
    
    bars = ax7.bar(range(len(thresholds)), success_rates, 
                   color=['darkgreen' if x > 80 else 'orange' if x > 50 else 'red' for x in success_rates],
                   alpha=0.8)
    
    ax7.set_xticks(range(len(thresholds)))
    ax7.set_xticklabels([f'<{t}' for t in thresholds])
    ax7.set_title('Başarı Oranları (Tüm Episode)', fontsize=14)
    ax7.set_xlabel('Hata Threshold')
    ax7.set_ylabel('Başarı Oranı (%)')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Bar'ların üzerine değerleri yaz
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 8. Kümülatif başarı oranı
    ax8 = axes[2, 1]
    
    # Her 50 adımda başarı oranını hesapla
    window_size = 50
    cumulative_success = []
    window_centers = []
    
    for i in range(window_size, len(position_errors), window_size//2):
        window_errors = position_errors[max(0, i-window_size):i]
        success_rate = (window_errors < 1.0).mean() * 100
        cumulative_success.append(success_rate)
        window_centers.append(i)
    
    if len(cumulative_success) > 0:
        ax8.plot(window_centers, cumulative_success, 'purple', linewidth=3, marker='o')
        ax8.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='%50 Başarı')
        ax8.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='%80 Başarı')
    
    ax8.set_title('Kümülatif Başarı Oranı (Threshold < 1.0)', fontsize=14)
    ax8.set_xlabel('Zaman Adımı')
    ax8.set_ylabel('Başarı Oranı (%)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Pozisyon dağılımı (heatmap)
    ax9 = axes[2, 2]
    
    # Agent ve target pozisyonlarının dağılımını göster
    ax9.scatter(target_positions[:, 0], target_positions[:, 1], 
               c='blue', alpha=0.3, s=20, label='Target Pozisyonları')
    ax9.scatter(agent_positions[:, 0], agent_positions[:, 1], 
               c='red', alpha=0.3, s=20, label='Agent Pozisyonları')
    
    # Başarılı noktaları büyük göster
    if len(success_indices) > 0:
        ax9.scatter(agent_positions[success_indices, 0], agent_positions[success_indices, 1], 
                   c='gold', s=100, marker='*', label=f'Başarılar ({len(success_indices)})', 
                   edgecolors='black', zorder=5)
    
    ax9.set_title('Pozisyon Dağılımı', fontsize=14)
    ax9.set_xlabel('X Pozisyonu')
    ax9.set_ylabel('Y Pozisyonu')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fixed_player14_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ✅ DETAYLI PERFORMANS RAPORU
    print("\n" + "="*80)
    print("🎯 DÜZELTILMIŞ PLAYER 14 AGENT PERFORMANS RAPORU")
    print("="*80)
    
    # Genel metrikler
    print(f"📊 Toplam test adımı: {len(position_errors)}")
    print(f"📍 Ortalama pozisyon hatası: {np.mean(position_errors):.2f}")
    print(f"📏 Standart sapma: {np.std(position_errors):.2f}")
    print(f"🎯 Minimum pozisyon hatası: {np.min(position_errors):.2f}")
    print(f"🎁 Ortalama reward: {np.mean(rewards):.2f}")
    print(f"⭐ Toplam başarı sayısı: {total_successes}")
    print(f"🏆 Genel başarı oranı: {(total_successes / len(position_errors) * 100):.1f}%")
    
    # Hareket analizi
    avg_predicted_speed = np.mean(np.linalg.norm(predicted_movements, axis=1))
    avg_real_speed = np.mean(np.linalg.norm(real_movements, axis=1))
    avg_movement_similarity = np.mean(movement_similarities)
    
    print(f"\n🚀 HAREKET ANALİZİ:")
    print(f"   • Agent ortalama hız: {avg_predicted_speed:.2f}")
    print(f"   • Target ortalama hız: {avg_real_speed:.2f}")
    print(f"   • Hız oranı: {(avg_predicted_speed/avg_real_speed*100):.1f}%")
    print(f"   • Hareket benzerliği: {avg_movement_similarity:.3f}")
    
    # Gelişim analizi
    first_third = len(position_errors) // 3
    last_third = len(position_errors) - first_third
    
    first_third_errors = position_errors[:first_third]
    last_third_errors = position_errors[last_third:]
    
    print(f"\n📈 GELİŞİM ANALİZİ:")
    print(f"   • İlk üçte bir ortalama hata: {np.mean(first_third_errors):.2f}")
    print(f"   • Son üçte bir ortalama hata: {np.mean(last_third_errors):.2f}")
    
    if np.mean(first_third_errors) > 0:
        improvement = ((np.mean(first_third_errors) - np.mean(last_third_errors)) / np.mean(first_third_errors) * 100)
        print(f"   • İyileşme oranı: {improvement:.1f}%")
        
        if improvement > 0:
            print("   ✅ Agent öğreniyor ve gelişiyor!")
        else:
            print("   ⚠️  Dikkat: Performans düşüyor, daha fazla eğitim gerekebilir")
    
    # Detaylı başarı analizi
    print(f"\n✅ DETAYLI BAŞARI ANALİZİ:")
    for i, threshold in enumerate(thresholds):
        print(f"   • {threshold} birimden az hata: {success_rates[i]:.1f}%")
    
    # Son dönem analizi
    final_quarter = len(position_errors) // 4
    final_errors = position_errors[-final_quarter:]
    final_success_rate = (final_errors < 1.0).mean() * 100
    
    print(f"\n🏁 SON DÖNEM ANALİZİ (Son %25):")
    print(f"   • Son dönem ortalama hata: {np.mean(final_errors):.2f}")
    print(f"   • Son dönem başarı oranı: {final_success_rate:.1f}%")
    
    # Genel değerlendirme
    best_success = max(success_rates[:3])  # İlk 3 threshold'da en iyi başarı
    print(f"\n🏆 GENEL DEĞERLENDİRME:")
    
    # Çoklu kritere göre değerlendirme
    criteria_score = 0
    
    # 1. Başarı oranı
    if best_success > 90:
        criteria_score += 25
        success_grade = "Mükemmel"
    elif best_success > 75:
        criteria_score += 20
        success_grade = "Çok İyi"
    elif best_success > 50:
        criteria_score += 15
        success_grade = "İyi"
    elif best_success > 25:
        criteria_score += 10
        success_grade = "Orta"
    else:
        criteria_score += 5
        success_grade = "Düşük"
    
    # 2. Hareket benzerliği
    if avg_movement_similarity > 0.7:
        criteria_score += 25
        movement_grade = "Mükemmel"
    elif avg_movement_similarity > 0.5:
        criteria_score += 20
        movement_grade = "Çok İyi"
    elif avg_movement_similarity > 0.3:
        criteria_score += 15
        movement_grade = "İyi"
    elif avg_movement_similarity > 0.1:
        criteria_score += 10
        movement_grade = "Orta"
    else:
        criteria_score += 5
        movement_grade = "Düşük"
    
    # 3. Gelişim
    if improvement > 20:
        criteria_score += 25
        improvement_grade = "Mükemmel"
    elif improvement > 10:
        criteria_score += 20
        improvement_grade = "Çok İyi"
    elif improvement > 0:
        criteria_score += 15
        improvement_grade = "İyi"
    elif improvement > -10:
        criteria_score += 10
        improvement_grade = "Orta"
    else:
        criteria_score += 5
        improvement_grade = "Düşük"
    
    # 4. Hareket aktivitesi
    if avg_predicted_speed > avg_real_speed * 0.8:
        criteria_score += 25
        activity_grade = "Mükemmel"
    elif avg_predicted_speed > avg_real_speed * 0.6:
        criteria_score += 20
        activity_grade = "Çok İyi"
    elif avg_predicted_speed > avg_real_speed * 0.4:
        criteria_score += 15
        activity_grade = "İyi"
    elif avg_predicted_speed > avg_real_speed * 0.2:
        criteria_score += 10
        activity_grade = "Orta"
    else:
        criteria_score += 5
        activity_grade = "Düşük"
    
    print(f"   📊 Başarı Oranı: {success_grade} ({best_success:.1f}%)")
    print(f"   🎯 Hareket Benzerliği: {movement_grade} ({avg_movement_similarity:.3f})")
    print(f"   📈 Gelişim: {improvement_grade} ({improvement:.1f}%)")
    print(f"   🚀 Hareket Aktivitesi: {activity_grade} ({avg_predicted_speed/avg_real_speed*100:.1f}%)")
    
    # Final grade
    if criteria_score >= 90:
        final_grade = "A+"
        evaluation = "🎉 MÜKEMMEL! Agent Player 14'ü neredeyse kusursuza yakın taklit ediyor!"
    elif criteria_score >= 80:
        final_grade = "A"
        evaluation = "🌟 HARIKA! Agent Player 14'ü çok başarılı şekilde öğrenmiş!"
    elif criteria_score >= 70:
        final_grade = "B+"
        evaluation = "👍 ÇOK İYİ! Agent Player 14'ü başarıyla taklit ediyor!"
    elif criteria_score >= 60:
        final_grade = "B"
        evaluation = "📈 İYİ! Agent öğreniyor ve gelişiyor!"
    elif criteria_score >= 50:
        final_grade = "C+"
        evaluation = "⚠️  ORTA! Daha fazla eğitim ve ayar gerekli."
    elif criteria_score >= 40:
        final_grade = "C"
        evaluation = "🔧 DÜŞÜK! Parametre ayarı ve uzun eğitim gerekli."
    else:
        final_grade = "D"
        evaluation = "❌ ÇOK DÜŞÜK! Temel parametreleri gözden geçirin."
    
    print(f"\n   🏆 TOPLAM PUAN: {criteria_score}/100")
    print(f"   📝 PERFORMANS NOTU: {final_grade}")
    print(f"   {evaluation}")
    
    # Öneriler
    print(f"\n💡 ÖNERİLER:")
    if criteria_score < 70:
        print("   • Daha uzun eğitim (300k+ timesteps)")
        print("   • Learning rate ayarı (daha düşük)")
        print("   • Reward fonksiyonu ince ayarı")
        print("   • Daha büyük network architecture")
    
    if avg_movement_similarity < 0.5:
        print("   • Movement reward ağırlığını artır")
        print("   • Action space'i gözden geçir")
        
    if improvement < 0:
        print("   • Overfitting olabilir, regularization ekle")
        print("   • Daha küçük learning rate dene")
    
    if avg_predicted_speed < avg_real_speed * 0.5:
        print("   • Movement bonus'u artır")
        print("   • Speed penalty'i azalt")
    
    print(f"\n🎊 Analiz tamamlandı! Sonuçlar 'fixed_player14_analysis.png' dosyasında.")
    
    return {
        'agent_positions': agent_positions,
        'target_positions': target_positions,
        'position_errors': position_errors,
        'movement_similarities': movement_similarities,
        'mean_error': np.mean(position_errors),
        'success_rate': total_successes / len(position_errors) * 100,
        'movement_similarity': avg_movement_similarity,
        'improvement': improvement,
        'final_grade': final_grade,
        'criteria_score': criteria_score
    }

if __name__ == "__main__":
    print("🚀 DÜZELTILMIŞ PLAYER 14 İMİTASYON PROJESİ BAŞLIYOR...")
    print("\n🔧 Yapılan Düzeltmeler:")
    print("✅ 1. Daha geniş ve realistik action space")
    print("✅ 2. Zenginleştirilmiş observation space")
    print("✅ 3. Hareket etmeyi teşvik eden reward sistemi")
    print("✅ 4. İlerleme takibi ve erken başarı tespiti")
    print("✅ 5. Geliştirilmiş eğitim parametreleri")
    print("✅ 6. Comprehensive değerlendirme sistemi")
    
    # Model eğitimi
    model, player14, ball = train_improved_agent()
    
    # Detaylı değerlendirme
    results = detailed_evaluation(model, player14, ball)
    
    print(f"\n🎊 Düzeltilmiş proje tamamlandı!")
    print(f"📊 Performans Notu: {results['final_grade']} ({results['criteria_score']}/100)")
    print(f"🎯 Başarı Oranı: {results['success_rate']:.1f}%")
    print(f"🚀 Hareket Benzerliği: {results['movement_similarity']:.3f}")
    print("📈 Detaylı analiz görselleştirmesi kaydedildi.")