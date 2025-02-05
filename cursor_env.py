import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2
import math
class CursorEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Action: Change in cursor position (dx, dy)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
        
        # Store previous action for smoothness calculation
        self.prev_action = None
        
        # Time tracking
        self.steps_since_last_target = 0
        self.time_penalty_factor = 0.1  # Penalty per step
        
        # Target stability tracking
        self.steps_in_target = 0
        self.required_stable_steps = 30  # Need to stay in target for 30 steps
        self.stability_bonus_base = 1.6  # Exponential base for stability reward
    
        # State parameters
        self.short_history_length = 50   # 50-step average for "current" gaze
        self.long_history_length = 500   # 500-step average for stable gaze
        self.gaze_history = np.zeros((self.long_history_length, 2))  # Store up to 500 positions
        
        # Observation: [50_step_avg_gaze_x, 50_step_avg_gaze_y, 500_step_avg_gaze_x, 500_step_avg_gaze_y, cursor_x, cursor_y]
        obs_dim = 6
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )
        
        # Screen dimensions
        self.width = 1920
        self.height = 1080
        
        # Movement parameters
        self.cursor_speed = 50
        self.target_switch_distance = 60  # Increased from 30 to 60 pixels
        
        # Gaze simulation parameters
        self.base_noise = 25.0  # Increased natural jitter
        self.drift_magnitude = 100.0  # Larger drifts
        self.drift_probability = 0.05  # More frequent drifts
        
        # Distraction simulation
        self.distraction_probability = 0.005  # More frequent distractions (0.5% per frame)
        self.distraction_duration = 0
        self.distraction_point = None
        self.returning_from_distraction = False
        self.return_speed = 0.15  # Slower return to target
        
        # Quick glance parameters (new)
        self.glance_probability = 0.02  # 2% chance per frame
        self.glance_radius_range = (100, 300)  # Quick glances 100-300 pixels away
        
        # Diagnostic tracking
        self.step_count = 0
        self.min_reward = float('inf')
        self.max_reward = float('-inf')
        self.min_distance = float('inf')
        self.max_distance = 0
        self.targets_reached = 0
        self.print_interval = 500
        
        # Initialize positions
        self.cursor_pos = np.array([self.width/2, self.height/2])
        self.target_pos = self.cursor_pos.copy()
        self.gaze_pos = self.target_pos.copy()
        
        self.render_mode = render_mode
        if render_mode == "human":
            cv2.namedWindow("Cursor Training", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Cursor Training", 960, 540)
    
    def _simulate_gaze(self):
        # Base position is target position
        target_gaze = self.target_pos.copy()
        
        # Add microsaccades and tremor (constant small noise)
        noise = np.random.normal(0, self.base_noise, 2)
        target_gaze += noise
        
        # Handle distractions and returns
        if self.distraction_duration > 0:
            # Currently distracted
            if self.distraction_duration == 1:
                # Last frame of distraction, start returning
                self.returning_from_distraction = True
            self.distraction_duration -= 1
            target_gaze = self.distraction_point
        elif self.returning_from_distraction:
            # Smoothly return from distraction
            to_target = self.target_pos - self.gaze_pos
            if np.linalg.norm(to_target) < 20:  # Close enough to target
                self.returning_from_distraction = False
            else:
                # Move gaze back towards target
                target_gaze = self.gaze_pos + to_target * self.return_speed
        elif np.random.random() < self.distraction_probability:
            # Start new distraction (longer duration)
            self.distraction_duration = np.random.randint(20, 60)  # 0.6-2.0 seconds
            # Generate random point to look at (might be off screen)
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(200, 800)  # Look further away
            self.distraction_point = self.target_pos + distance * np.array([np.cos(angle), np.sin(angle)])
            target_gaze = self.distraction_point
        elif np.random.random() < self.glance_probability:
            # Quick glance somewhere nearby (no duration, just instant movement)
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(*self.glance_radius_range)
            target_gaze = self.target_pos + distance * np.array([np.cos(angle), np.sin(angle)])
        else:
            # Regular drift
            if np.random.random() < self.drift_probability:
                drift = np.random.normal(0, self.drift_magnitude, 2)
                target_gaze += drift
        
        # Smooth movement towards target_gaze
        to_target_gaze = target_gaze - self.gaze_pos
        self.gaze_pos += to_target_gaze * 0.3  # Smooth follow
        
        # Clip to screen (with larger margin for realism)
        margin = 200  # Increased margin
        self.gaze_pos = np.clip(self.gaze_pos, 
                               [-margin, -margin], 
                               [self.width + margin, self.height + margin])
    
    def _get_obs(self):
        # Update gaze history (shift everything back and add new position)
        self.gaze_history[1:] = self.gaze_history[:-1]
        self.gaze_history[0] = self.gaze_pos / [self.width, self.height]
        
        # Calculate short-term (50-step) average
        short_avg = np.mean(self.gaze_history[:self.short_history_length], axis=0)
        
        # Calculate long-term (500-step) average
        long_avg = np.mean(self.gaze_history, axis=0)
        
        # Return short-term average, long-term average, and cursor position
        return np.concatenate([
            short_avg,                                  # 50-step average gaze
            long_avg,                                  # 500-step average gaze
            self.cursor_pos / [self.width, self.height] # Cursor position
        ]).astype(np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset previous action and stability tracking
        self.prev_action = None
        self.steps_in_target = 0
        
        # Print stats if interval reached
        print(f"\nStats after {self.step_count} steps:")
        print(f"Targets reached: {self.targets_reached}")
        print(f"Reward range: [{self.min_reward:.2f}, {self.max_reward:.2f}]")
        print(f"Distance range: [{self.min_distance:.1f}, {self.max_distance:.1f}]")
        print(f"Current cursor pos: [{self.cursor_pos[0]:.1f}, {self.cursor_pos[1]:.1f}]")
        print(f"Current target pos: [{self.target_pos[0]:.1f}, {self.target_pos[1]:.1f}]")
        
        # Random target position (not too close to edges)
        self.target_pos = np.array([
            np.random.uniform(100, self.width-100),
            np.random.uniform(100, self.height-100)
        ])
        
        # Start cursor at random position
        self.cursor_pos = np.array([
            np.random.uniform(0, self.width),
            np.random.uniform(0, self.height)
        ])
        
        # Initialize gaze at target
        self.gaze_pos = self.target_pos.copy()
        
        # Reset gaze history with current normalized gaze position
        normalized_gaze = self.gaze_pos / [self.width, self.height]
        self.gaze_history = np.tile(normalized_gaze, (self.long_history_length, 1))
        
        # Simulate first gaze position
        self._simulate_gaze()
        
        return self._get_obs(), {}
    
    def step(self, action):
        self.step_count += 1
        
        # Store current action for next step
        self.prev_action = action.copy()

        # Update cursor position based on action
        delta = action * self.cursor_speed
        self.cursor_pos += delta
        self.cursor_pos = np.clip(self.cursor_pos, [0, 0], [self.width, self.height])
        
        # Simulate realistic gaze
        self._simulate_gaze()
        
        # Calculate distance to target
        distance = np.linalg.norm(self.cursor_pos - self.target_pos)
        
        # Update distance stats
        self.min_distance = min(self.min_distance, distance)
        self.max_distance = max(self.max_distance, distance)
        
        # Base distance reward (inverse of distance)
        reward = 100000 / (distance + 10e-3)
        
        # Check if cursor is within target radius
        in_target = distance < self.target_switch_distance
        
        if in_target:
            self.steps_in_target += 1
            # Exponential bonus for maintaining position in target
            stability_bonus = pow(self.stability_bonus_base, self.steps_in_target)
            reward += stability_bonus
        else:
            self.steps_in_target = 0
        
        # Update reward stats
        self.min_reward = min(self.min_reward, reward)
        self.max_reward = max(self.max_reward, reward)
        
        # Episode terminates when stable in target for required steps
        done = self.steps_in_target >= self.required_stable_steps
        if done:
            reward += 100000  # Final completion bonus
            self.targets_reached += 1
            # Reset gaze history with current normalized gaze position for new target
            normalized_gaze = self.gaze_pos / [self.width, self.height]
            self.gaze_history = np.tile(normalized_gaze, (self.long_history_length, 1))
        
        if self.render_mode == "human":
            self.render()

        print(f"Reward: {reward:.2f}, Steps in target: {self.steps_in_target}")
        
        return self._get_obs(), reward, done, False, {}
    
    def render(self):
        # Create black background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw target (green) - increased circle size to match new radius
        cv2.circle(frame, self.target_pos.astype(int), 40, (0, 255, 0), -1)
        
        # Draw gaze position (red)
        cv2.circle(frame, self.gaze_pos.astype(int), 15, (0, 0, 255), 2)
        
        # Draw cursor (blue)
        cv2.circle(frame, self.cursor_pos.astype(int), 10, (255, 0, 0), -1)
        
        cv2.imshow("Cursor Training", frame)
        cv2.waitKey(1)
    
    def close(self):
        if self.render_mode == "human":
            cv2.destroyAllWindows()