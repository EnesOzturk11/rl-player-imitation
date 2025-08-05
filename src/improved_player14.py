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
        
        # DAHA KÜÇÜK VE REALİSTİK ACTION SPACE
        # Player 14'ün gerçek hareketlerine bakalım
        pos = self.player14_data[["position_x", "position_y"]].values
        if len(pos) > 1:
            dx = np.diff(pos[:, 0])
            dy = np.diff(pos[:, 1])
            speeds = np.sqrt(dx**2 + dy**2)
            
            # İstatistikler
            max_speed = np.percentile(speeds, 95)  # %95'lik dilimi kullan (outlier'ları ignore et)
            mean_speed = np.mean(speeds)
            std_speed = np.std(speeds)
            
            print(f"Player 14 max hız (95th percentile): {max_speed:.2f}")
            print(f"Player 14 ortalama hız: {mean_speed:.2f}")
            print(f"Player 14 hız std: {std_speed:.2f}")
            
            # Action space'i daha makul sınırlarla tanımla
            speed_limit = min(max_speed * 1.2, mean_speed + 3 * std_speed)  # Daha conservative
        else:
            speed_limit = 5.0
        
        print(f"Action space hız limiti: {speed_limit:.2f}")
        
        # Action Space: Daha küçük ve realistic
        self.action_space = spaces.Box(
            low=np.array([-speed_limit, -speed_limit], dtype=np.float32), 
            high=np.array([speed_limit, speed_limit], dtype=np.float32),
            dtype=np.float32
        )
        
        # DAHA ZENGİN OBSERVATION SPACE
        # [agent_x, agent_y, ball_x, ball_y, ball_direction, 
        #  relative_ball_x, relative_ball_y, prev_dx, prev_dy, target_next_x, target_next_y]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        
        self.current_step = 0
        self.agent_pos = None
        self.prev_movement = np.array([0.0, 0.0])
        
        # Normalizasyon için scale faktörleri
        self.pos_scale = 1.0 / np.std(pos.flatten()) if len(pos) > 0 else 1.0
        print(f"Pozisyon scale faktörü: {self.pos_scale:.4f}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        
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
        """Zengin gözlem alanı oluştur"""
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
        
        # Göreceli ball pozisyonu (agent'a göre)
        relative_ball = ball_pos - self.agent_pos
        
        # Bir sonraki target pozisyonu (eğer varsa)
        if self.current_step + 1 < len(self.player14_data):
            next_target = self.player14_data[["position_x", "position_y"]].iloc[self.current_step + 1].values
        else:
            next_target = self.player14_data[["position_x", "position_y"]].iloc[-1].values
        
        next_target = np.nan_to_num(next_target, nan=0.0)
        
        # Observation vektörü: [agent_x, agent_y, ball_x, ball_y, ball_direction, 
        #                       relative_ball_x, relative_ball_y, prev_dx, prev_dy, 
        #                       target_next_x, target_next_y]
        observation = np.concatenate([
            self.agent_pos * self.pos_scale,  # Normalize edilmiş pozisyon
            ball_pos * self.pos_scale,        # Normalize edilmiş ball pozisyonu
            ball_dir / 360.0,                 # Normalize edilmiş açı
            relative_ball * self.pos_scale,   # Göreceli pozisyon
            self.prev_movement,               # Önceki hareket
            next_target * self.pos_scale      # Hedef pozisyon
        ])
        
        return np.nan_to_num(observation, nan=0.0).astype(np.float32)

    def step(self, action):
        # Action'ı temizle ve sınırla
        action = np.nan_to_num(action, nan=0.0)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Agent'ı hareket ettir
        self.agent_pos += action
        self.prev_movement = action.copy()
        self.current_step += 1
        
        # Player 14'ün bu adımdaki gerçek pozisyonu
        if self.current_step < len(self.player14_data):
            target_pos = self.player14_data[["position_x", "position_y"]].iloc[self.current_step].values
            
            # Player 14'ün gerçek hareketi
            prev_target_pos = self.player14_data[["position_x", "position_y"]].iloc[self.current_step-1].values
            real_movement = target_pos - prev_target_pos
        else:
            target_pos = self.player14_data[["position_x", "position_y"]].iloc[-1].values
            real_movement = np.array([0.0, 0.0])
        
        # NaN kontrolü
        target_pos = np.nan_to_num(target_pos, nan=0.0)
        self.agent_pos = np.nan_to_num(self.agent_pos, nan=0.0)
        real_movement = np.nan_to_num(real_movement, nan=0.0)
        
        # GELİŞTİRİLMİŞ REWARD SİSTEMİ
        
        # 1. Pozisyon hatası (normalize edilmiş)
        position_error = np.linalg.norm(self.agent_pos - target_pos)
        position_reward = -position_error * self.pos_scale  # Normalize edilmiş hata
        
        # 2. Hareket yönü benzerliği 
        movement_error = np.linalg.norm(action - real_movement)
        movement_reward = -movement_error * 0.5
        
        # 3. Hız penaltısı (çok hızlı hareket etmeyi engellemek için)
        speed_penalty = -np.linalg.norm(action) * 0.1
        
        # 4. Smooth movement reward (ani değişiklikleri engellemek için)
        if hasattr(self, 'prev_action'):
            smoothness_penalty = -np.linalg.norm(action - self.prev_action) * 0.2
        else:
            smoothness_penalty = 0.0
        
        self.prev_action = action.copy()
        
        # 5. Pozisyon proximity bonus (hedefe yaklaştıkça bonus)
        proximity_bonus = max(0, 50 - position_error) * 0.1
        
        # Toplam reward
        total_reward = (position_reward + movement_reward + speed_penalty + 
                       smoothness_penalty + proximity_bonus)
        
        # Reward'ı makul sınırlarda tut
        total_reward = np.clip(total_reward, -100, 100)
        
        # Episode bitme koşulları
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Yeni gözlem
        observation = self._get_observation()
        
        info = {
            'step': self.current_step,
            'position_error': position_error,
            'movement_error': movement_error,
            'target_pos': target_pos,
            'agent_pos': self.agent_pos.copy(),
            'real_movement': real_movement,
            'predicted_movement': action,
            'position_reward': position_reward,
            'movement_reward': movement_reward,
            'proximity_bonus': proximity_bonus
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
    
    print("✓ Veriler temizlendi")
    return ball, player14

def make_env(player14, ball):
    """Environment wrapper"""
    def _init():
        return ImprovedPlayer14Env(player14, ball)
    return _init

def train_improved_agent():
    """Geliştirilmiş agent eğitimi"""
    print("=== GELİŞTİRİLMİŞ PLAYER 14 AGENT EĞİTİMİ ===")
    
    # Veri yükleme
    ball, player14 = load_and_clean_data()
    
    # Environment oluşturma
    env = DummyVecEnv([make_env(player14, ball)])
    
    # GELİŞTİRİLMİŞ MODEL PARAMETRELERİ
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,      # Daha düşük learning rate
        n_steps=2048,            # Daha büyük batch
        batch_size=128,          # Daha büyük batch size
        n_epochs=10,             # Daha az epoch (overfitting'i önlemek için)
        gamma=0.99,              # Daha yüksek discount (uzun vadeli düşünmek için)
        gae_lambda=0.95,
        clip_range=0.2,          # Standart clip range
        ent_coef=0.01,           # Exploration için entropy coefficient
        vf_coef=0.5,             # Value function coefficient
        max_grad_norm=0.5,       # Gradient clipping
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128, 64], vf=[128, 128, 64])]  # Daha büyük network
        ),
        tensorboard_log="./tensorboard_logs_improved/"
    )
    
    # Evaluation callback
    eval_env = DummyVecEnv([make_env(player14, ball)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models_improved/",
        log_path="./logs_improved/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Eğitim
    print("🚀 Geliştirilmiş eğitim başlıyor...")
    model.learn(
        total_timesteps=100000,  # Daha uzun eğitim
        callback=eval_callback,
        progress_bar=False
    )
    
    # Model kaydetme
    model.save("improved_player14_model")
    print("✅ Geliştirilmiş model kaydedildi: improved_player14_model.zip")
    
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
    movement_errors = []
    predicted_movements = []
    real_movements = []
    
    # Test uzunluğu
    test_length = min(len(player14) - 1, 400)
    
    print(f"Test uzunluğu: {test_length} adım")
    
    for step in range(test_length):
        # Model tahmini
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Sonuçları kaydet
        agent_positions.append(info['agent_pos'].copy())
        target_positions.append(info['target_pos'].copy())
        rewards.append(reward)
        position_errors.append(info['position_error'])
        movement_errors.append(info['movement_error'])
        predicted_movements.append(info['predicted_movement'].copy())
        real_movements.append(info['real_movement'].copy())
        
        if terminated or truncated:
            print(f"Episode {step+1} adımda bitti")
            break
    
    # NumPy arrays'e çevir
    agent_positions = np.array(agent_positions)
    target_positions = np.array(target_positions)
    position_errors = np.array(position_errors)
    movement_errors = np.array(movement_errors)
    predicted_movements = np.array(predicted_movements)
    real_movements = np.array(real_movements)
    
    # GÖRSELLEŞTIRME
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('GELİŞTİRİLMİŞ Player 14 Agent Performans Analizi', fontsize=18, fontweight='bold')
    
    # 1. Tam yörünge karşılaştırması
    ax1 = axes[0, 0]
    ax1.plot(target_positions[:, 0], target_positions[:, 1], 'b-', 
             linewidth=4, label='Gerçek Player 14', alpha=0.9)
    ax1.plot(agent_positions[:, 0], agent_positions[:, 1], 'r--', 
             linewidth=3, label='AI Agent', alpha=0.8)
    ax1.scatter(target_positions[0, 0], target_positions[0, 1], 
                c='green', s=150, marker='o', label='Başlangıç', zorder=5)
    ax1.scatter(target_positions[-1, 0], target_positions[-1, 1], 
                c='red', s=150, marker='x', label='Bitiş', zorder=5)
    ax1.set_title('Tam Yörünge Karşılaştırması', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Pozisyonu')
    ax1.set_ylabel('Y Pozisyonu')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Son 80 adımın detaylı görünümü
    ax2 = axes[0, 1]
    last_steps = min(80, len(target_positions))
    ax2.plot(target_positions[-last_steps:, 0], target_positions[-last_steps:, 1], 
             'b-', linewidth=5, label='Gerçek Player 14', alpha=0.9)
    ax2.plot(agent_positions[-last_steps:, 0], agent_positions[-last_steps:, 1], 
             'r--', linewidth=4, label='AI Agent', alpha=0.8)
    
    # Son 20 noktayı vurgula
    for i in range(-min(20, last_steps), 0):
        ax2.scatter(target_positions[i, 0], target_positions[i, 1], 
                   c='blue', s=80, alpha=0.8, edgecolors='darkblue')
        ax2.scatter(agent_positions[i, 0], agent_positions[i, 1], 
                   c='red', s=80, alpha=0.8, edgecolors='darkred')
    
    ax2.set_title(f'SON {last_steps} ADIM - YAKINSAMA ANALİZİ', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X Pozisyonu')
    ax2.set_ylabel('Y Pozisyonu')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Pozisyon hatası zaman grafiği (geliştirilmiş)
    ax3 = axes[0, 2]
    
    # Moving average hesapla
    window = 20
    if len(position_errors) > window:
        moving_avg = np.convolve(position_errors, np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(position_errors)), moving_avg, 'purple', 
                linewidth=4, label=f'{window}-adım ortalaması')
    
    ax3.plot(position_errors, 'lightgreen', alpha=0.6, linewidth=1, label='Ham hata')
    ax3.axhline(y=np.mean(position_errors), color='r', linestyle='--', 
                linewidth=2, label=f'Genel ortalama: {np.mean(position_errors):.1f}')
    
    # Son kısım vurgusu
    if len(position_errors) > 100:
        ax3.fill_between(range(len(position_errors)-100, len(position_errors)), 
                        0, max(position_errors), alpha=0.15, color='orange',
                        label='Son 100 adım')
    
    ax3.set_title('Pozisyon Hatası Evrimi', fontsize=14)
    ax3.set_xlabel('Zaman Adımı')
    ax3.set_ylabel('Mesafe Hatası')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Hareket karşılaştırması (geliştirilmiş)
    ax4 = axes[1, 0]
    pred_magnitudes = np.linalg.norm(predicted_movements, axis=1)
    real_magnitudes = np.linalg.norm(real_movements, axis=1)
    
    # Moving averages
    if len(pred_magnitudes) > window:
        pred_smooth = np.convolve(pred_magnitudes, np.ones(window)/window, mode='valid')
        real_smooth = np.convolve(real_magnitudes, np.ones(window)/window, mode='valid')
        ax4.plot(range(window-1, len(pred_magnitudes)), pred_smooth, 'r-', 
                linewidth=3, label='AI Agent (smooth)', alpha=0.9)
        ax4.plot(range(window-1, len(real_magnitudes)), real_smooth, 'b-', 
                linewidth=3, label='Gerçek Player 14 (smooth)', alpha=0.9)
    
    ax4.plot(real_magnitudes, 'lightblue', alpha=0.4, linewidth=1, label='Gerçek (ham)')
    ax4.plot(pred_magnitudes, 'lightcoral', alpha=0.4, linewidth=1, label='AI (ham)')
    
    ax4.set_title('Hareket Hızı Karşılaştırması', fontsize=14)
    ax4.set_xlabel('Zaman Adımı')
    ax4.set_ylabel('Hareket Hızı')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Reward evrimi (geliştirilmiş)
    ax5 = axes[1, 1]
    
    # Moving average
    if len(rewards) > window:
        reward_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax5.plot(range(window-1, len(rewards)), reward_smooth, 'purple', 
                linewidth=4, label=f'{window}-adım ortalaması')
    
    ax5.plot(rewards, 'lightgray', alpha=0.5, linewidth=1, label='Ham reward')
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax5.axhline(y=np.mean(rewards), color='red', linestyle='--', 
                linewidth=2, label=f'Ortalama: {np.mean(rewards):.1f}')
    
    ax5.set_title('Reward Evrimi', fontsize=14)
    ax5.set_xlabel('Zaman Adımı')
    ax5.set_ylabel('Reward')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Başarı analizi
    ax6 = axes[1, 2]
    
    # Farklı threshold'lar için başarı oranları
    thresholds = [1, 2, 5, 10, 20, 50]
    success_rates = []
    
    last_errors = position_errors[-100:] if len(position_errors) > 100 else position_errors
    
    for threshold in thresholds:
        success_rate = (last_errors < threshold).mean() * 100
        success_rates.append(success_rate)
    
    bars = ax6.bar(range(len(thresholds)), success_rates, 
                   color=['darkgreen' if x > 70 else 'orange' if x > 50 else 'red' for x in success_rates],
                   alpha=0.8)
    
    ax6.set_xticks(range(len(thresholds)))
    ax6.set_xticklabels([f'<{t}' for t in thresholds])
    ax6.set_title('Son 100 Adımda Başarı Oranları', fontsize=14)
    ax6.set_xlabel('Hata Threshold')
    ax6.set_ylabel('Başarı Oranı (%)')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Bar'ların üzerine değerleri yaz
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('improved_player14_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # PERFORMANS RAPORU
    print("\n" + "="*60)
    print("🎯 GELİŞTİRİLMİŞ PLAYER 14 AGENT PERFORMANS RAPORU")
    print("="*60)
    
    # Genel metrikler
    print(f"📊 Toplam test adımı: {len(position_errors)}")
    print(f"📍 Ortalama pozisyon hatası: {np.mean(position_errors):.2f}")
    print(f"📏 Standart sapma: {np.std(position_errors):.2f}")
    print(f"🎁 Ortalama reward: {np.mean(rewards):.2f}")
    print(f"🎁 Son 100 adım ortalama reward: {np.mean(rewards[-100:]):.2f}")
    
    # Gelişim analizi
    first_100_errors = position_errors[:100] if len(position_errors) > 100 else position_errors[:len(position_errors)//2]
    last_100_errors = position_errors[-100:] if len(position_errors) > 100 else position_errors[len(position_errors)//2:]
    
    print(f"\n🔥 GELİŞİM ANALİZİ:")
    print(f"   • İlk 100 adım ortalama hata: {np.mean(first_100_errors):.2f}")
    print(f"   • Son 100 adım ortalama hata: {np.mean(last_100_errors):.2f}")
    
    if np.mean(first_100_errors) > 0:
        improvement = ((np.mean(first_100_errors) - np.mean(last_100_errors)) / np.mean(first_100_errors) * 100)
        print(f"   • İyileşme oranı: {improvement:.1f}%")
    
    # Detaylı başarı analizi
    print(f"\n✅ DETAYLI BAŞARI ANALİZİ (Son 100 adım):")
    for i, threshold in enumerate(thresholds):
        print(f"   • {threshold} birimden az hata: {success_rates[i]:.1f}%")
    
    # Genel değerlendirme
    best_success = max(success_rates[:3])  # İlk 3 threshold'da en iyi başarı
    print(f"\n🏆 GENEL DEĞERLENDİRME:")
    if best_success > 80:
        print("   🎉 MÜKEMMEL! Agent Player 14'ü çok başarılı taklit ediyor!")
        grade = "A+"
    elif best_success > 60:
        print("   👍 ÇOK İYİ! Agent Player 14'ü başarılı şekilde öğrenmiş!")
        grade = "A"
    elif best_success > 40:
        print("   📈 İYİ! Agent öğreniyor ve gelişiyor!")
        grade = "B"
    elif best_success > 20:
        print("   ⚠️  ORTA! Daha fazla eğitim gerekebilir.")
        grade = "C"
    else:
        print("   ❌ DÜŞÜK! Parametre ayarı ve daha uzun eğitim gerekli.")
        grade = "D"
    
    print(f"   📝 Performans Notu: {grade}")
    
    return {
        'agent_positions': agent_positions,
        'target_positions': target_positions,
        'position_errors': position_errors,
        'movement_errors': movement_errors,
        'mean_error': np.mean(position_errors),
        'last_100_mean_error': np.mean(last_100_errors),
        'success_rates': success_rates,
        'grade': grade
    }

if __name__ == "__main__":
    print("🚀 GELİŞTİRİLMİŞ PLAYER 14 İMİTASYON PROJESİ BAŞLIYOR...")
    
    # Model eğitimi
    model, player14, ball = train_improved_agent()
    
    # Detaylı değerlendirme
    results = detailed_evaluation(model, player14, ball)
    
    print(f"\n🎊 Geliştirilmiş proje tamamlandı! Performans notu: {results['grade']}")
    print("📊 Sonuçlar 'improved_player14_analysis.png' dosyasına kaydedildi.")