from stable_baselines3 import PPO
from cursor_env import CursorEnv
import cv2
import os

def main():
    print("\nTraining cursor control with simulated gaze data")
    print("Window shows:")
    print("- Green circle: Target to reach")
    print("- Red circle: Simulated gaze (mostly on target with natural jitter)")
    print("- Blue circle: Cursor learning to follow gaze")
    print("\nPress 'q' to quit, Ctrl+C to save and exit")
    
    # Create environment
    env = CursorEnv(render_mode="human")
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.003,  # Slightly higher learning rate
        n_steps=2048,  # Collect more steps before updating
        batch_size=64,
        n_epochs=10,
        gamma=0.99,  # Focus on long-term reward
        verbose=1
    )
    
    # Load existing model if available
    model_path = "cursor_ppo"
    if os.path.exists(model_path + ".zip"):
        print("Loading existing model...")
        model = PPO.load(model_path, env=env)
    
    try:
        # Train continuously
        model.learn(total_timesteps=int(1e6), progress_bar=True)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted, saving model...")
    
    finally:
        # Save model
        model.save(model_path)
        env.close()
        print("Model saved to", model_path)

if __name__ == "__main__":
    main()