assuming you have docker installed...
1. clone github repo
2. cd into project
3. run below commands

```
> docker build -t streetfighterai -f Dockerfile .
> docker run -p 8888:8888 streetfighterai
```

4. to open in browser, check your terminal and click the boxed link

![image](https://github.com/user-attachments/assets/4b9c6dc8-4deb-44d8-a76b-745e765c8425)

5. to train a new agent, open `train.ipynb`. to run the models described in our paper, open `run_developed_models.ipynb`

**note:** to see the agent actually play, you need to run the python notebook locally with Python 3.8.8 (requirements.txt file included). Running on the browser, you cannot see the agent play visually (but you can see the steps it took and whatnot)

### environments
all environments are described below
note that for `run_developed_models.ipynb`, the environment still has to match even when loading from zip, so change `class StreetFighter` with environments below. the names of the models match with the name of the environment

###### BEST
```python
 # Best Model ENV
class StreetFighter(Env): # pass in basic env from above to preprocessing
    def __init__(self):
        super().__init__() # inherit from base env
        # Specify action space and observation space 
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8) # grayscaled frame, smaller amt of pixels
        self.action_space = MultiBinary(12) # type of actions that can be taken
        # Startup and instance of the game 
        self.game = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', use_restricted_actions=retro.Actions.FILTERED) # used to get valid button combos
    
    def reset(self): # restart
        # Return the first frame 
        obs = self.game.reset()
        obs = self.preprocess(obs) 
        self.previous_frame = obs # sets previous frame to current frame
        
        # Create a attribute to hold the score delta 
        self.score = 0 
        return obs
    
    def preprocess(self, observation): # grayscale, resize
        # Grayscaling 
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        # Resize 
        resize = cv2.resize(gray, (84,84), interpolation=cv2.INTER_CUBIC)
        # Add the channels value
        channels = np.reshape(resize, (84,84,1))
        return channels 
    
    def step(self, action): # how do we process action
        # Take a step 
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs) 
        
        # Frame delta 
        frame_delta = obs - self.previous_frame # change in pixels
        self.previous_frame = obs 
        
        # Reshape the reward function
        reward = info['score'] - self.score 
        self.score = info['score'] 
        
        return frame_delta, reward, done, info
    
    def render(self, *args, **kwargs): # unpack any args and kwargs from stable baseline
        self.game.render()
        
    def close(self):
        self.game.close()
```

###### reward1
```python
# 1st Reward Func
class StreetFighter(Env):
    def __init__(self):
        super().__init__()
        # Constants for reward calculation
        self.START_HEALTH = 176  # Starting health in Street Fighter II
        self.ROUND_WIN_BONUS = 500
        self.PERFECT_WIN_BONUS = 1000
        
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(84, 84, 3),  # Keep 3 channels for RGB
            dtype=np.uint8
        )
        self.action_space = MultiBinary(12)  # type of actions that can be taken
        self.game = retro.make(
            game='StreetFighterIISpecialChampionEdition-Genesis',
            use_restricted_actions=retro.Actions.FILTERED
        )
        
        # Initialize health tracking
        self.enemy_health = self.START_HEALTH
        self.agent_health = self.START_HEALTH
        self.score = 0

    def reset(self):
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        
        # Reset game state variables
        self.score = 0
        self.enemy_health = self.START_HEALTH
        self.agent_health = self.START_HEALTH
        
        return obs
    
    def preprocess(self, observation):
        # Resize first to reduce computation
        resized = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Simple color quantization using bitwise operations
        # Reduce to 3 bits per channel (8 values per channel)
        quantized = resized & 0b11100000
        
        # Optional: Create more distinct colors by increasing contrast
        # This helps make different elements more distinguishable
        quantized = cv2.convertScaleAbs(quantized, alpha=1.2, beta=10)
        
        return quantized
        
    def calculate_reward(self, info, enemy_damage_taken, agent_damage_taken):
        reward = 0
        
        # 1. Base damage reward/penalty with dynamic scaling
        damage_diff = enemy_damage_taken - agent_damage_taken
        if damage_diff > 0:
            reward += damage_diff * (1 + (self.agent_health / self.START_HEALTH))
        else:
            reward += damage_diff * (1 + (self.enemy_health / self.START_HEALTH))

        # 4. Round outcome rewards
        if info['enemy_health'] <= 0:  # Victory
            win_reward = self.ROUND_WIN_BONUS
            if self.agent_health == self.START_HEALTH:
                win_reward += self.PERFECT_WIN_BONUS
            health_ratio = self.agent_health / self.START_HEALTH
            reward += win_reward * health_ratio
        elif info['health'] <= 0:  # Loss
            loss_penalty = -self.ROUND_WIN_BONUS * (info['enemy_health'] / self.START_HEALTH)
            reward += loss_penalty
        
        return reward
    
    def step(self, action):
        obs, _, done, info = self.game.step(action)
        obs = self.preprocess(obs)
        
        enemy_damage_taken = abs(info['enemy_health'] - self.enemy_health)
        agent_damage_taken = abs(info['health'] - self.agent_health)
        
        reward = self.calculate_reward(info, enemy_damage_taken, agent_damage_taken)
        
        # Update health tracking
        self.enemy_health = info['enemy_health']
        self.agent_health = info['health']
        
        return obs, reward, done, info
    
    def render(self, *args, **kwargs):
        self.game.render()

    def close(self):
        self.game.close()
```

###### reward2
```python
# 2nd Reward Func
class StreetFighter(Env):
    def __init__(self):
        super().__init__()
        # Constants for reward calculation
        self.START_HEALTH = 176  # Starting health in Street Fighter II
        self.ROUND_WIN_MULTIPLIER = 2
        self.ROUND_LOSS_MULTIPLIER = -1
        
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(84, 84, 3),  # Keep 3 channels for RGB
            dtype=np.uint8
        )
        self.action_space = MultiBinary(12)  # type of actions that can be taken
        self.game = retro.make(
            game='StreetFighterIISpecialChampionEdition-Genesis',
            use_restricted_actions=retro.Actions.FILTERED
        )
        
        # Initialize health tracking
        self.enemy_health = self.START_HEALTH
        self.agent_health = self.START_HEALTH
        self.score = 0

    def reset(self):
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        
        # Reset game state variables
        self.score = 0
        self.enemy_health = self.START_HEALTH
        self.agent_health = self.START_HEALTH
        
        return obs
    
    def preprocess(self, observation):
        # Resize first to reduce computation
        resized = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Simple color quantization using bitwise operations
        # Reduce to 3 bits per channel (8 values per channel)
        quantized = resized & 0b11100000
        
        # Optional: Create more distinct colors by increasing contrast
        # This helps make different elements more distinguishable
        quantized = cv2.convertScaleAbs(quantized, alpha=1.2, beta=10)
        
        return quantized
        
    def calculate_reward(self, info):
        reward = 0

        # 1. Score reward
        reward += (info['score'] - self.score) * .1

        # 2. Round outcome rewards with health
        if info['enemy_health'] <= 0:  # Victory
            health_ratio = self.agent_health / self.START_HEALTH
            reward += self.ROUND_WIN_MULTIPLIER * health_ratio
        elif info['health'] <= 0:  # Loss
            health_ratio = (info['enemy_health'] / self.START_HEALTH)
            reward += self.ROUND_LOSS_MULTIPLIER * health_ratio

        return reward
    
    def step(self, action):
        obs, _, done, info = self.game.step(action)
        obs = self.preprocess(obs)
        
        reward = self.calculate_reward(info)
        
        # Update health tracking
        self.enemy_health = info['enemy_health']
        self.agent_health = info['health']
        self.score = info['score']
        
        return obs, reward, done, info
    
    def render(self, *args, **kwargs):
        self.game.render()

    def close(self):
        self.game.close()
```

###### reg
```python
# Regular Runs (Mult Bin)
class StreetFighter(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(84, 84, 3),  # Keep 3 channels for RGB
            dtype=np.uint8
        )
        self.action_space = Multibinary(12)  # type of actions that can be taken
        self.game = retro.make(
            game='StreetFighterIISpecialChampionEdition-Genesis',
            use_restricted_actions=retro.Actions.FILTERED
        )

    def reset(self):
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        self.score = 0
        return obs

    def preprocess(self, observation):
        # Resize first to reduce computation
        resized = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Simple color quantization using bitwise operations
        # Reduce to 3 bits per channel (8 values per channel)
        quantized = resized & 0b11100000
        
        # Optional: Create more distinct colors by increasing contrast
        # This helps make different elements more distinguishable
        quantized = cv2.convertScaleAbs(quantized, alpha=1.2, beta=10)
        
        # Method 1: Simple bitwise quantization
        return quantized

    def step(self, action):
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs)
        reward = info['score'] - self.score
        self.score = info['score']
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        self.game.render()

    def close(self):
        self.game.close()
```

###### reg (DQN ONLY)
```python
# OG DISC Model (Regular Runs)
class StreetFighter(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(84, 84, 3),  # Keep 3 channels for RGB
            dtype=np.uint8
        )
        self.action_space = Discrete(12)  # type of actions that can be taken
        self.game = retro.make(
            game='StreetFighterIISpecialChampionEdition-Genesis',
            use_restricted_actions=retro.Actions.DISCRETE
        )

    def reset(self):
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        self.score = 0
        return obs

    def preprocess(self, observation):
        # Resize first to reduce computation
        resized = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Simple color quantization using bitwise operations
        # Reduce to 3 bits per channel (8 values per channel)
        quantized = resized & 0b11100000
        
        # Optional: Create more distinct colors by increasing contrast
        # This helps make different elements more distinguishable
        quantized = cv2.convertScaleAbs(quantized, alpha=1.2, beta=10)
        
        # Method 1: Simple bitwise quantization
        return quantized

    def step(self, action):
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs)
        reward = info['score'] - self.score
        self.score = info['score']
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        self.game.render()

    def close(self):
        self.game.close()
```