import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
from env import PlayerImitationEnv

# Veri yükleme
def load_data():
    ball = pd.read_csv("data/ball_clean.csv")
    player_1 = pd.read_csv("data/tracker_1_clean.csv")
    player_2 = pd.read_csv("data/tracker_2_clean.csv")
    player_14 = pd.read_csv("data/tracker_14_clean.csv")
    player_15 = pd.read_csv("data/tracker_15_clean.csv")
    
    # Veri temizleme - NaN ve infinity değerleri temizle
    def clean_dataframe(df):
        # NaN değerleri forward fill ile doldur, sonra 0 ile
        df = df.fillna(method='ffill').fillna(0)
        # Infinity değerleri 0 ile değiştir
        df = df.replace([np.inf, -np.inf], 0)
        return df
    
    ball = clean_dataframe(ball)
    player_1 = clean_dataframe(player_1)
    player_2 = clean_dataframe(player_2)
    player_14 = clean_dataframe(player_14)
    player_15 = clean_dataframe(player_15)
    
    players = {
        "Player 1": player_1,
        "Player 2": player_2,
        "Player 14": player_14,
        "Player 15": player_15,
    }
    
    return ball, players

# En uygun oyuncuyu seç (en dinamik harekete sahip olan)
def select_best_player(players):
    max_movement = 0
    best_player = None
    best_name = None
    
    for name, player_data in players.items():
        # Toplam hareket mesafesini hesapla
        pos = player_data[["position_x", "position_y"]].values
        movements = np.sum(np.sqrt(np.sum(np.diff(pos, axis=0)**2, axis=1)))
        print(f"{name}: Total movement = {movements:.2f}")
        
        if movements > max_movement:
            max_movement = movements
            best_player = player_data
            best_name = name
    
    print(f"Selected player: {best_name}")
    return best_player, best_name

# Environment wrapper
def make_env(target_player, ball):
    def _init():
        return PlayerImitationEnv(target_player, ball)
    return _init

def train_model():
    # Veri yükleme
    ball, players = load_data()
    target_player, player_name = select_best_player(players)
    
    # Environment oluşturma
    env = DummyVecEnv([make_env(target_player, ball)])
    
    # Model oluşturma (PPO algoritması)
    model = PPO(
        "MlpPolicy",  # Multi-Layer Perceptron policy
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./tensorboard_logs/"
    )
    
    # Evaluation callback
    eval_env = DummyVecEnv([make_env(target_player, ball)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Model eğitimi
    print("Eğitim başlıyor...")
    model.learn(
        total_timesteps=100000,  # Toplam eğitim adımı
        callback=eval_callback,
        progress_bar=False  # Progress bar'ı kapatıldı
    )
    
    # Modeli kaydet
    model.save(f"player_imitation_model_{player_name.replace(' ', '_').lower()}")
    print("Model kaydedildi.")
    
    return model, target_player, ball, player_name

def evaluate_model(model, target_player, ball, player_name):
    """Modeli değerlendir ve sonuçları görselleştir"""
    
    # Test environment
    env = PlayerImitationEnv(target_player, ball)
    
    # Model ile tahmin yap
    obs, _ = env.reset()
    agent_positions = []
    target_positions = []
    rewards = []
    
    for step in range(min(len(target_player) - 1, 200)):  # İlk 200 adım için test
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        agent_positions.append(info['agent_pos'].copy())
        target_positions.append(info['target_pos'].copy())
        rewards.append(reward)
        
        if terminated or truncated:
            break
    
    agent_positions = np.array(agent_positions)
    target_positions = np.array(target_positions)
    
    # Görselleştirme
    plt.figure(figsize=(15, 5))
    
    # Pozisyon karşılaştırması
    plt.subplot(1, 3, 1)
    plt.plot(target_positions[:, 0], target_positions[:, 1], 'b-', label='Gerçek Oyuncu', linewidth=2)
    plt.plot(agent_positions[:, 0], agent_positions[:, 1], 'r--', label='AI Agent', linewidth=2)
    plt.title('Pozisyon Karşılaştırması')
    plt.xlabel('X Pozisyonu')
    plt.ylabel('Y Pozisyonu')
    plt.legend()
    plt.grid(True)
    
    # Hata analizi
    plt.subplot(1, 3, 2)
    errors = np.linalg.norm(agent_positions - target_positions, axis=1)
    plt.plot(errors)
    plt.title('Pozisyon Hatası (Zaman)')
    plt.xlabel('Zaman Adımı')
    plt.ylabel('Mesafe Hatası')
    plt.grid(True)
    
    # Reward analizi
    plt.subplot(1, 3, 3)
    plt.plot(rewards)
    plt.title('Reward Değişimi')
    plt.xlabel('Zaman Adımı')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'evaluation_results_{player_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Metrikleri yazdır
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    mean_reward = np.mean(rewards)
    
    print(f"\n=== Değerlendirme Sonuçları ({player_name}) ===")
    print(f"Ortalama Pozisyon Hatası: {mean_error:.3f}")
    print(f"Standart Sapma: {std_error:.3f}")
    print(f"Ortalama Reward: {mean_reward:.3f}")
    print(f"Toplam Test Adımı: {len(errors)}")
    
    return {
        'mean_error': mean_error,
        'std_error': std_error,
        'mean_reward': mean_reward,
        'agent_positions': agent_positions,
        'target_positions': target_positions
    }

if __name__ == "__main__":
    # Model eğitimi
    model, target_player, ball, player_name = train_model()
    
    # Değerlendirme
    results = evaluate_model(model, target_player, ball, player_name)
    
    print("Eğitim ve değerlendirme tamamlandı!")