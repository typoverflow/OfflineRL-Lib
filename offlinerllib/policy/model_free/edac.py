import torch
import torch.optim as optim
from operator import itemgetter

from typing import Union, Tuple, Dict

from offlinerllib.policy.model_free.sacn import SACNPolicy
from offlinerllib.module.actor import BaseActor
from offlinerllib.module.critic import Critic


class EDACPolicy(SACNPolicy):
    """
    Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble <Ref:https://arxiv.org/abs/2110.01548>
    """

    def __init__(
        self,
        actor: BaseActor,
        critic: Critic,
        actor_optim: optim.Optimizer,
        critic_optim: optim.Optimizer,
        tau: float = 0.005,
        eta: float = 1.0,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, float]] = 0.2,
        do_reverse_update: bool = False,
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__(actor, critic, actor_optim, critic_optim, tau, gamma,
                         alpha, do_reverse_update, device)
        self.eta = eta

    def _critic_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        critic_loss, critic_loss_metrics = super()._critic_loss(batch)
        obss, actions, next_obss, rewards, terminals = itemgetter("observations", "actions", "next_observations", "rewards", "terminals")(batch)
        ensemble_size = self.critic.ensemble_size
        assert (ensemble_size > 1)
        diversity_loss = None
        
        obss = obss.unsqueeze(0).repeat_interleave(ensemble_size, dim=0)
        actions = actions.unsqueeze(0).repeat_interleave(ensemble_size, dim=0).requires_grad_(True)
        q_values = self.critic(obss, actions)
        raw_grad = torch.autograd.grad(q_values.sum(), actions, retain_graph=True, create_graph=True)[0]
        normalized_grad = raw_grad / (torch.norm(raw_grad, p=2, dim=2).unsqueeze(-1) + 1e-10)
        
        # [ensemble_size, batch_size, action_dim] -> [batch_size, ensemble_size, action_dim]
        normalized_grad = normalized_grad.transpose(0, 1)
        grad_prod = normalized_grad @ normalized_grad.permute(0, 2, 1)
        
        # mask shape as [batch_size, 1, 1]
        masks = (torch.eye(ensemble_size, device=self.device).unsqueeze(dim=0).repeat(grad_prod.shape[0], 1, 1))
        
        diversity_loss = (1-masks) * grad_prod
        diversity_loss = diversity_loss.sum(dim=(1, 2)).mean()
        scale = ensemble_size - 1
        diversity_loss = diversity_loss / scale
        
        critic_loss = critic_loss + self.eta * diversity_loss
        critic_loss_metrics["loss/critic_diversity_loss"] = diversity_loss.item()
        return (critic_loss, critic_loss_metrics)