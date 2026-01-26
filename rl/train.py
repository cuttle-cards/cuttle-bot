"""Train RL agent for Cuttle game using MaskablePPO with true self-play."""
from __future__ import annotations
import os

import numpy as np
import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.monitor import Monitor

from rl.config import LOG_DIR, MODEL_DIR, TRAINING_CONFIG
from rl.cuttle_env import CuttleRLEnvironment
from rl.self_play_env import AdaptiveSelfPlayWrapper


class ActivationLogger:
    """Capture policy activations for TensorBoard logging."""

    def __init__(self, policy: torch.nn.Module) -> None:
        self._policy = policy
        self._activations: dict[str, torch.Tensor] = {}
        self._handles = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name, module in self._policy.named_modules():
            if isinstance(module, torch.nn.Linear):
                handle = module.register_forward_hook(self._make_hook(name))
                self._handles.append(handle)

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            self._activations[name] = output.detach().cpu()

        return hook

    def clear(self) -> None:
        self._activations.clear()

    def get(self) -> dict[str, torch.Tensor]:
        return self._activations

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class DiagnosticsCallback(BaseCallback):
    """Log action stats, masks, and activations to TensorBoard."""

    def __init__(self, log_freq: int = 1000, activation_freq: int = 5000) -> None:
        super().__init__()
        self.log_freq = log_freq
        self.activation_freq = activation_freq
        self._tb_writer = None
        self._activation_logger: ActivationLogger | None = None

    def _on_training_start(self) -> None:
        for fmt in self.logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                self._tb_writer = fmt.writer
                break
        self._activation_logger = ActivationLogger(self.model.policy)

    def _on_training_end(self) -> None:
        if self._activation_logger:
            self._activation_logger.close()

    def _get_action_mask(self) -> np.ndarray | None:
        if not hasattr(self.training_env, "envs"):
            return None
        base_env = self.training_env.envs[0]
        try:
            return base_env.unwrapped.action_masks()
        except Exception:
            return None

    def _log_activations(self, obs: np.ndarray) -> None:
        if not self._tb_writer or not self._activation_logger:
            return
        self._activation_logger.clear()
        with torch.no_grad():
            obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
            self.model.policy(obs_tensor)
        for name, activation in self._activation_logger.get().items():
            self._tb_writer.add_histogram(
                f"activations/{name}",
                activation,
                self.num_timesteps,
            )

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True

        def to_numpy(data):
            if torch.is_tensor(data):
                return data.detach().cpu().numpy()
            return np.asarray(data)

        actions = self.locals.get("actions")
        if actions is not None:
            actions_np = to_numpy(actions).flatten()
            self.logger.record("rollout/action_mean", float(np.mean(actions_np)))
            if self._tb_writer:
                self._tb_writer.add_histogram(
                    "actions/selected",
                    actions_np,
                    self.num_timesteps,
                )

        values = self.locals.get("values")
        if values is not None:
            values_np = to_numpy(values)
            self.logger.record("rollout/value_mean", float(np.mean(values_np)))

        mask = self._get_action_mask()
        if mask is not None:
            self.logger.record("rollout/legal_action_count", float(mask.sum()))
            self.logger.record("rollout/legal_action_fraction", float(mask.mean()))
            if self._tb_writer:
                self._tb_writer.add_histogram(
                    "actions/mask",
                    mask.astype(np.int32),
                    self.num_timesteps,
                )

        if self.n_calls % self.activation_freq == 0:
            obs = self.locals.get("new_obs")
            if obs is None:
                obs = self.locals.get("obs")
            if obs is not None:
                self._log_activations(obs)

        return True


class SelfPlayCallback(BaseCallback):
    """Callback to update opponent model during training for true self-play."""
    
    def __init__(self, self_play_env: AdaptiveSelfPlayWrapper, update_freq: int = 10000):
        super().__init__()
        self.self_play_env = self_play_env
        self.update_freq = update_freq
        self._last_update = 0
        
    def _on_training_start(self) -> None:
        # Set initial opponent model
        self.self_play_env.set_opponent_model(self.model)
        print("🎮 Self-play initialized: opponent will gradually use trained model")
        
    def _on_step(self) -> bool:
        # Update opponent model periodically
        if self.num_timesteps - self._last_update >= self.update_freq:
            self.self_play_env.set_opponent_model(self.model)
            self._last_update = self.num_timesteps
            
            # Log current model usage probability
            prob = self.self_play_env._get_model_probability()
            self.logger.record("self_play/model_prob", prob)
            print(f"📊 Self-play update @ {self.num_timesteps}: opponent model prob = {prob:.1%}")
            
        return True


def mask_fn(env):
    """Function that returns action mask for MaskablePPO."""
    # Unwrap to get to the actual environment with action_masks method
    while hasattr(env, 'env'):
        if hasattr(env, 'action_masks'):
            return env.action_masks()
        env = env.env
    return env.action_masks()


def main():
    """Main training function with true self-play."""
    # Large action spaces can trip strict simplex validation in torch distributions.
    torch.distributions.Distribution.set_default_validate_args(False)

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print("Initializing environment with action masking and self-play...")
    
    # Create base environment with adaptive self-play
    # Strategy: Start random-only, gradually introduce model opponent
    # This ensures agent learns to win before facing harder opponents
    base_env = CuttleRLEnvironment()
    self_play_env = AdaptiveSelfPlayWrapper(
        base_env,
        model_prob_start=0.0,    # Start with 100% random opponent
        model_prob_end=0.3,       # End with only 30% model (mostly random for wins)
        transition_steps=300000,  # Very slow transition over 300K steps
    )
    
    # Wrap with Monitor and ActionMasker
    env = Monitor(self_play_env, LOG_DIR)
    env = ActionMasker(env, mask_fn)  # Critical: wrap with ActionMasker
    
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
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=MODEL_DIR,
        name_prefix="cuttle_rl",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    diagnostics_callback = DiagnosticsCallback()
    self_play_callback = SelfPlayCallback(
        self_play_env,
        update_freq=10000,  # Update opponent model every 10K steps
    )
    
    # Train the model
    print(f"Starting training for {TRAINING_CONFIG['total_timesteps']} timesteps...")
    print("Using action masking - model will only consider legal actions!")
    print("Using adaptive self-play - opponent gradually uses trained model!")
    print("Progress will be shown below. This may take 15-30 minutes.")
    
    model.learn(
        total_timesteps=TRAINING_CONFIG["total_timesteps"],
        callback=[checkpoint_callback, diagnostics_callback, self_play_callback],
        progress_bar=False,
    )
    
    # Save final model
    final_model_path = os.path.join(MODEL_DIR, "cuttle_rl_final")
    model.save(final_model_path)
    
    print(f"\nTraining complete!")
    print(f"Final model saved to: {final_model_path}.zip")
    print(f"View training logs with: make tensorboard")


if __name__ == "__main__":
    main()
