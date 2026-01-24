"""Train RL agent for Cuttle game using MaskablePPO."""
import os

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from rl.config import LOG_DIR, MODEL_DIR, TRAINING_CONFIG
from rl.cuttle_env import CuttleRLEnvironment
from rl.self_play_env import SelfPlayWrapper


def main():
    """Main training function."""
    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print("Initializing environment with action masking...")
    # Create and wrap environment
    env = CuttleRLEnvironment()
    env = SelfPlayWrapper(env)
    env = Monitor(env, LOG_DIR)
    
    print("Creating MaskablePPO model...")
    # Create MaskablePPO model (supports action masking)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        n_steps=TRAINING_CONFIG["n_steps"],
        batch_size=TRAINING_CONFIG["batch_size"],
        n_epochs=TRAINING_CONFIG["n_epochs"],
        gamma=TRAINING_CONFIG["gamma"],
        gae_lambda=TRAINING_CONFIG["gae_lambda"],
        clip_range=TRAINING_CONFIG["clip_range"],
        ent_coef=TRAINING_CONFIG["ent_coef"],
        verbose=TRAINING_CONFIG["verbose"],
        tensorboard_log=LOG_DIR,
    )
    
    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=MODEL_DIR,
        name_prefix="cuttle_rl",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # Train the model
    print(f"Starting training for {TRAINING_CONFIG['total_timesteps']} timesteps...")
    print("Using action masking - model will only consider legal actions!")
    print("Progress will be shown below. This may take 15-30 minutes.")
    
    model.learn(
        total_timesteps=TRAINING_CONFIG["total_timesteps"],
        callback=checkpoint_callback,
        progress_bar=True,
    )
    
    # Save final model
    final_model_path = os.path.join(MODEL_DIR, "cuttle_rl_final")
    model.save(final_model_path)
    
    print(f"\nTraining complete!")
    print(f"Final model saved to: {final_model_path}.zip")
    print(f"View training logs with: make tensorboard")


if __name__ == "__main__":
    main()
