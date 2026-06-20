import numpy as np
from typing import List, Any, Optional, Dict
from src.dsl import Hypothesis

class ARC3Agent:
    """
    Agent for ARC-AGI-3 interactive environments.
    Maintains a world model, forms hypotheses about goals,
    and chooses actions that maximise info gain about the win condition.
    """

    def __init__(self, max_steps: int = 30):
        self.max_steps = max_steps
        self.hypotheses: List[Hypothesis] = []
        self.world_model: dict = {}
        self.action_history: List[Dict[str, Any]] = []
        self.observation_history: List[dict] = []
        self.goal_hypothesis: Optional[str] = None

    def observe(self, obs: dict):
        """Process new environment observation."""
        self.observation_history.append(obs)
        self._update_world_model(obs)

        # First observation: generate initial goal hypotheses
        if len(self.observation_history) == 1:
            self.hypotheses = self._seed_goal_hypotheses(obs)

    def _seed_goal_hypotheses(self, obs: dict) -> List[Hypothesis]:
        """From first observation, generate candidate win conditions."""
        hyps = []
        goal_templates = [
            "reach_specific_cell",
            "match_target_pattern",
            "collect_all_objects_of_color",
            "sort_objects_by_attribute",
            "create_symmetry",
            "fill_empty_region",
            "replicate_pattern_n_times",
        ]
        for i, template in enumerate(goal_templates):
            hyps.append(Hypothesis(
                id=f"goal_{i:02d}",
                description=template,
                condition="always",
                transform=template,
                confidence=1.0/len(goal_templates)
            ))
        return hyps

    def _update_world_model(self, obs: dict):
        """Update internal world model from observation."""
        self.world_model["last_obs"] = obs
        self.world_model["step"] = len(self.observation_history)

        if "reward" in obs:
            self.world_model["last_reward"] = obs["reward"]
        if "done" in obs:
            self.world_model["done"] = obs["done"]

    def select_action(self, available_actions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Choose action with highest expected info gain about goal hypothesis.
        First 1/3 of steps strictly maximize information gain (Explore First).
        """
        active_hyps = [h for h in self.hypotheses if not h.falsified]

        if not active_hyps or len(available_actions) == 0:
            import random
            return random.choice(available_actions) if available_actions else None

        current_step = len(self.action_history)
        explore_only = current_step < (self.max_steps / 3)

        scored = []
        for action in available_actions[:20]:  # cap for compute
            predicted_reward = self._predict_reward(action, active_hyps)
            info_gain_score  = self._action_info_gain(action, active_hyps)
            
            if explore_only:
                score = info_gain_score
            else:
                confidence = max(h.confidence for h in active_hyps)
                score = (confidence * predicted_reward +
                         (1 - confidence) * info_gain_score)
            scored.append((score, action))

        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[0][1]

    def _predict_reward(self, action: Dict[str, Any], hyps: List[Hypothesis]) -> float:
        """Weighted average predicted reward under hypothesis ensemble."""
        total_weight = sum(h.confidence for h in hyps)
        if total_weight == 0: return 0.0
        return sum(h.confidence * self._hyp_action_score(h, action)
                   for h in hyps) / total_weight

    def _hyp_action_score(self, hyp: Hypothesis, action: Dict[str, Any]) -> float:
        """How well does this action align with this hypothesis goal?"""
        if "reach" in hyp.description and action.get("type") == "move":
            return 0.6
        if "fill" in hyp.description and action.get("type") == "select":
            return 0.6
        return 0.3

    def _action_info_gain(self, action: Dict[str, Any], hyps: List[Hypothesis]) -> float:
        """How much info does this action give us?"""
        if action not in self.action_history[-5:]:
            return 0.8
        return 0.2

    def update_after_step(self, action: Dict[str, Any], new_obs: dict):
        """Update beliefs after receiving environment feedback."""
        reward = new_obs.get("reward", 0.0)
        self.action_history.append(action)

        for hyp in self.hypotheses:
            if hyp.falsified: continue
            expected = self._hyp_action_score(hyp, action)
            if reward > 0.5 and expected < 0.3:
                hyp.confidence *= 1.5
            elif reward < 0.1 and expected > 0.7:
                hyp.confidence *= 0.5
                if hyp.confidence < 0.05:
                    hyp.falsified = True

        self.observe(new_obs)
