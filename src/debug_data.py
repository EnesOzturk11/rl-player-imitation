import pandas as pd
import numpy as np

def check_data_quality():
    """Veri kalitesini kontrol et"""
    print("=== VERİ KALİTE KONTROLÜ ===")
    
    # Veri yükleme
    ball = pd.read_csv("data/ball_clean.csv")
    player_2 = pd.read_csv("data/tracker_2_clean.csv")
    
    print("\n--- BALL VERİSİ ---")
    print(f"Shape: {ball.shape}")
    print(f"Columns: {ball.columns.tolist()}")
    print(f"NaN değerler:")
    print(ball.isnull().sum())
    print(f"\nİlk 5 satır:")
    print(ball.head())
    print(f"\nSon 5 satır:")
    print(ball.tail())
    
    print("\n--- PLAYER 2 VERİSİ ---")
    print(f"Shape: {player_2.shape}")
    print(f"Columns: {player_2.columns.tolist()}")
    print(f"NaN değerler:")
    print(player_2.isnull().sum())
    print(f"\nİlk 5 satır:")
    print(player_2.head())
    print(f"\nSon 5 satır:")
    print(player_2.tail())
    
    # Infinity kontrolü
    print("\n--- INFINITY KONTROLÜ ---")
    for col in ball.select_dtypes(include=[np.number]).columns:
        inf_count = np.isinf(ball[col]).sum()
        if inf_count > 0:
            print(f"Ball {col}: {inf_count} infinity değer")
    
    for col in player_2.select_dtypes(include=[np.number]).columns:
        inf_count = np.isinf(player_2[col]).sum()
        if inf_count > 0:
            print(f"Player 2 {col}: {inf_count} infinity değer")
    
    # Veri boyutları uyumlu mu?
    print(f"\n--- BOYUT KONTROLÜ ---")
    print(f"Ball veri uzunluğu: {len(ball)}")
    print(f"Player 2 veri uzunluğu: {len(player_2)}")
    
    # Min-max değerleri
    print(f"\n--- DEĞİŞKEN ARALIĞI ---")
    print("Ball:")
    print(ball.describe())
    print("\nPlayer 2:")
    print(player_2.describe())
    
    return ball, player_2

def test_environment():
    """Environment'ı test et"""
    from env import PlayerImitationEnv
    
    print("\n=== ENVIRONMENT TEST ===")
    
    ball, player_2 = check_data_quality()
    
    # NaN ve infinity değerleri temizle
    ball = ball.fillna(method='ffill').fillna(0)
    player_2 = player_2.fillna(method='ffill').fillna(0)
    
    # Infinity değerleri clip et
    ball = ball.replace([np.inf, -np.inf], 0)
    player_2 = player_2.replace([np.inf, -np.inf], 0)
    
    try:
        env = PlayerImitationEnv(player_2, ball)
        print("✓ Environment başarıyla oluşturuldu")
        
        # Reset test
        obs, info = env.reset()
        print(f"✓ Reset başarılı. Observation shape: {obs.shape}")
        print(f"  Observation: {obs}")
        print(f"  NaN var mı: {np.isnan(obs).any()}")
        print(f"  Inf var mı: {np.isinf(obs).any()}")
        
        # Step test
        action = env.action_space.sample()
        print(f"  Test action: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Step başarılı")
        print(f"  New observation: {obs}")
        print(f"  Reward: {reward}")
        print(f"  NaN var mı: {np.isnan(obs).any()}")
        print(f"  Inf var mı: {np.isinf(obs).any()}")
        
        # Birkaç adım daha test et
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if np.isnan(obs).any() or np.isinf(obs).any():
                print(f"❌ Adım {i+2}'de NaN/Inf bulundu!")
                print(f"   Observation: {obs}")
                break
            else:
                print(f"✓ Adım {i+2} başarılı - Reward: {reward:.3f}")
        
    except Exception as e:
        print(f"❌ Environment hatası: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_environment()