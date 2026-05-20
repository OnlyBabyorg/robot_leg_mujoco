import gymnasium as gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class AsymmetricExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        self.proprio_dim = observation_space["proprio"].shape[0]
        self.priv_dim = observation_space["privileged"].shape[0]

        
        self.actor_net = th.nn.Sequential(
            th.nn.Linear(self.proprio_dim, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 128),
            th.nn.ReLU(),
        )
       
        self.critic_net = th.nn.Sequential(
            th.nn.Linear(self.proprio_dim + self.priv_dim, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 128),
            th.nn.ReLU(),
        )
        self.actor_out = th.nn.Linear(128, features_dim)
        self.critic_out = th.nn.Linear(128, features_dim)

    def forward(self, observations: dict) -> th.Tensor:
        propro = observations["proprio"]
        return self.actor_out(self.actor_net(propro))

    def forward_critic(self, observations: dict) -> th.Tensor:
        propro = observations["proprio"]
        priv = observations["privileged"]
        combined = th.cat([propro, priv], dim=1)
        return self.critic_out(self.critic_net(combined))