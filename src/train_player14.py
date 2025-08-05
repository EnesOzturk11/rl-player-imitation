import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
from player14_env import Player14ImitationEnv

def load_and_clean_data():
    """Player 14 ve ball verisini yükle ve temizle"""
    print("=== VERİ YÜKLEME ===")
    
    ball = pd.read_csv("data/ball_clean.csv")
    player14 = pd.read_csv("data/tracker_14_clean.csv")
    
    print(f"Ball veri boyutu: {ball.shape}")
    print(f"Player 14 veri boyutu: {player14.shape}")
    
    # Veri temizleme
    def clean_dataframe(df):
        df = df.fillna(method='ffill').fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        return df
    
    ball = clean_dataframe(ball)
    player14 = clean_dataframe(player14)
    
    print("✓ Veriler temizlendi")
    return ball, player14

def make_env(player14, ball):
    """Environment wrapper"""
    def _init():
        return Player14ImitationEnv(player14, ball)
    return _init

def train_player14_agent():
    """Player 14 için özel agent eğitimi"""
    print("=== PLAYER 14 AGENT EĞİTİMİ ===")
    
    # Veri yükleme
    ball, player14 = load_and_clean_data()
    
    # Environment oluşturma
    env = DummyVecEnv([make_env(player14, ball)])
    
    # Model parametreleri (Player 14'e özel optimizasyon)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-4,  # Biraz daha yüksek learning rate
        n_steps=1024,  # Daha küçük batch
        batch_size=32,
        n_epochs=20,  # Daha fazla epoch
        gamma=0.95,  # Biraz daha az discount
        gae_lambda=0.9,
        clip_range=0.1,  # Daha küçük clip
        tensorboard_log="./tensorboard_logs_player14/"
    )
    
    # Evaluation callback
    eval_env = DummyVecEnv([make_env(player14, ball)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models_player14/",
        log_path="./logs_player14/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Eğitim
    print("🚀 Eğitim başlıyor...")
    model.learn(
        total_timesteps=75000,  # Player 14 için optimize edilmiş timestep
        callback=eval_callback,
        progress_bar=False
    )
    
    # Model kaydetme
    model.save("player14_imitation_model")
    print("✅ Model kaydedildi: player14_imitation_model.zip")
    
    return model, player14, ball

def detailed_evaluation(model, player14, ball):
    """Detaylı değerlendirme ve görselleştirme"""
    print("\n=== DETAYLI DEĞERLENDİRME ===")
    
    # Test environment
    env = Player14ImitationEnv(player14, ball)
    
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
    
    # Test uzunluğu (son hareketlerin benzerliğini görmek için yeterince uzun)
    test_length = min(len(player14) - 1, 300)
    
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
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Player 14 Agent Performans Analizi', fontsize=16, fontweight='bold')
    
    # 1. Tam yörünge karşılaştırması
    ax1 = axes[0, 0]
    ax1.plot(target_positions[:, 0], target_positions[:, 1], 'b-', 
             linewidth=3, label='Gerçek Player 14', alpha=0.8)
    ax1.plot(agent_positions[:, 0], agent_positions[:, 1], 'r--', 
             linewidth=2, label='AI Agent', alpha=0.9)
    ax1.scatter(target_positions[0, 0], target_positions[0, 1], 
                c='green', s=100, marker='o', label='Başlangıç', zorder=5)
    ax1.scatter(target_positions[-1, 0], target_positions[-1, 1], 
                c='red', s=100, marker='x', label='Bitiş', zorder=5)
    ax1.set_title('Tam Yörünge Karşılaştırması')
    ax1.set_xlabel('X Pozisyonu')
    ax1.set_ylabel('Y Pozisyonu')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Son 50 adımın detaylı görünümü (RL'in başarısını görmek için)
    ax2 = axes[0, 1]
    last_steps = min(50, len(target_positions))
    ax2.plot(target_positions[-last_steps:, 0], target_positions[-last_steps:, 1], 
             'b-', linewidth=4, label='Gerçek Player 14', alpha=0.8)
    ax2.plot(agent_positions[-last_steps:, 0], agent_positions[-last_steps:, 1], 
             'r--', linewidth=3, label='AI Agent', alpha=0.9)
    
    # Son noktaları vurgula
    for i in range(-min(10, last_steps), 0):
        ax2.scatter(target_positions[i, 0], target_positions[i, 1], 
                   c='blue', s=50, alpha=0.7)
        ax2.scatter(agent_positions[i, 0], agent_positions[i, 1], 
                   c='red', s=50, alpha=0.7)
    
    ax2.set_title(f'SON {last_steps} ADIM (RL Başarı Göstergesi)', fontweight='bold')
    ax2.set_xlabel('X Pozisyonu')
    ax2.set_ylabel('Y Pozisyonu')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Pozisyon hatası zaman grafiği
    ax3 = axes[0, 2]
    ax3.plot(position_errors, 'g-', linewidth=2)
    ax3.axhline(y=np.mean(position_errors), color='r', linestyle='--', 
                label=f'Ortalama: {np.mean(position_errors):.2f}')
    
    # Son 50 adımı vurgula
    if len(position_errors) > 50:
        ax3.fill_between(range(len(position_errors)-50, len(position_errors)), 
                        0, max(position_errors), alpha=0.2, color='yellow',
                        label='Son 50 adım')
    
    ax3.set_title('Pozisyon Hatası (Zaman)')
    ax3.set_xlabel('Zaman Adımı')
    ax3.set_ylabel('Mesafe Hatası')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Hareket yönü karşılaştırması
    ax4 = axes[1, 0]
    # Hareket vektörlerinin büyüklüğü
    pred_magnitudes = np.linalg.norm(predicted_movements, axis=1)
    real_magnitudes = np.linalg.norm(real_movements, axis=1)
    
    ax4.plot(real_magnitudes, 'b-', linewidth=2, label='Gerçek Hareket Hızı', alpha=0.8)
    ax4.plot(pred_magnitudes, 'r--', linewidth=2, label='Tahmin Edilen Hız', alpha=0.8)
    ax4.set_title('Hareket Hızı Karşılaştırması')
    ax4.set_xlabel('Zaman Adımı')
    ax4.set_ylabel('Hareket Hızı')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Reward evrimi
    ax5 = axes[1, 1]
    # Moving average hesapla
    window = 20
    if len(rewards) > window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax5.plot(range(window-1, len(rewards)), moving_avg, 'purple', 
                linewidth=3, label=f'{window}-adım ortalaması')
    
    ax5.plot(rewards, 'gray', alpha=0.5, linewidth=1, label='Ham reward')
    ax5.set_title('Reward Evrimi')
    ax5.set_xlabel('Zaman Adımı')
    ax5.set_ylabel('Reward')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Son adımlarda hata dağılımı (histogram)
    ax6 = axes[1, 2]
    last_errors = position_errors[-50:] if len(position_errors) > 50 else position_errors
    ax6.hist(last_errors, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax6.axvline(x=np.mean(last_errors), color='red', linestyle='--', 
                linewidth=2, label=f'Ortalama: {np.mean(last_errors):.2f}')
    ax6.set_title('Son Adımlarda Hata Dağılımı')
    ax6.set_xlabel('Pozisyon Hatası')
    ax6.set_ylabel('Frekans')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('player14_agent_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # PERFORMANS METRİKLERİ
    print("\n" + "="*50)
    print("🎯 PLAYER 14 AGENT PERFORMANS RAPORU")
    print("="*50)
    
    # Genel metrikler
    print(f"📊 Toplam test adımı: {len(position_errors)}")
    print(f"📍 Ortalama pozisyon hatası: {np.mean(position_errors):.3f}")
    print(f"📏 Standart sapma: {np.std(position_errors):.3f}")
    print(f"🎁 Ortalama reward: {np.mean(rewards):.2f}")
    
    # Son adımların analizi (RL başarısının göstergesi)
    last_50_errors = position_errors[-50:] if len(position_errors) > 50 else position_errors
    first_50_errors = position_errors[:50] if len(position_errors) > 50 else position_errors
    
    print(f"\n🔥 SON 50 ADIM ANALİZİ (RL Başarı Göstergesi):")
    print(f"   • Son 50 adım ortalama hata: {np.mean(last_50_errors):.3f}")
    print(f"   • İlk 50 adım ortalama hata: {np.mean(first_50_errors):.3f}")
    print(f"   • İyileşme oranı: {((np.mean(first_50_errors) - np.mean(last_50_errors)) / np.mean(first_50_errors) * 100):.1f}%")
    
    # Hareket benzerliği
    print(f"\n🏃 HAREKET ANALİZİ:")
    print(f"   • Ortalama hareket hatası: {np.mean(movement_errors):.3f}")
    
    # Başarı değerlendirmesi
    success_threshold = 5.0  # Bu değer veri setinize göre ayarlanabilir
    success_rate = (last_50_errors < success_threshold).mean() * 100
    print(f"\n✅ BAŞARI DEĞERLENDİRMESİ:")
    print(f"   • Son 50 adımda {success_threshold} birimden az hata oranı: {success_rate:.1f}%")
    
    if success_rate > 70:
        print("   🎉 MÜKEMMEL! Agent Player 14'ü çok başarılı taklit ediyor!")
    elif success_rate > 50:
        print("   👍 İYİ! Agent Player 14'ü başarılı şekilde öğrenmiş!")
    elif success_rate > 30:
        print("   📈 ORTA! Agent öğreniyor ama daha eğitim gerekebilir.")
    else:
        print("   ⚠️  DÜŞÜK! Daha fazla eğitim veya parametre ayarı gerekli.")
    
    return {
        'agent_positions': agent_positions,
        'target_positions': target_positions,
        'position_errors': position_errors,
        'movement_errors': movement_errors,
        'mean_error': np.mean(position_errors),
        'last_50_mean_error': np.mean(last_50_errors),
        'success_rate': success_rate
    }

if __name__ == "__main__":
    print("🚀 PLAYER 14 İMİTASYON PROJESİ BAŞLIYOR...")
    
    # Model eğitimi
    model, player14, ball = train_player14_agent()
    
    # Detaylı değerlendirme
    results = detailed_evaluation(model, player14, ball)
    
    print("\n🎊 Proje tamamlandı! Sonuçlar 'player14_agent_analysis.png' dosyasına kaydedildi.")