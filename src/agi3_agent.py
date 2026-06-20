import numpy as np
from typing import List, Any, Optional
from src.dsl import Hypothesis

class ARC3Agent:
    """
    Agent for ARC-AGI-3 interactive environments.
    Maintains a world model, forms hypotheses about goals,
    and chooses actions that maximise info gain about the win condition.
    """

    def __init__(self):
        self.hypotheses: List[Hypothesis] = []
        self.world_model: dict = {}
        self.action_history: List[Any] = []
        self.observation_history: List[Any] = []
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
        grid = np.array(obs.get("grid", []))

        # Hypothesis classes for interactive envs
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

    def select_action(self, available_actions: List[Any]) -> Any:
        """
        Choose action with highest expected info gain about goal hypothesis.
        Falls back to random if no clear signal.
        """
        active_hyps = [h for h in self.hypotheses if not h.falsified]

        if not active_hyps or len(available_actions) == 0:
            import random
            return random.choice(available_actions) if available_actions else None

        # Score each action by expected disambiguation power
        scored = []
        for action in available_actions[:20]:  # cap for compute
            # Simulate expected outcome (simplified)
            predicted_reward = self._predict_reward(action, active_hyps)
            info_gain_score  = self._action_info_gain(action, active_hyps)
            # Balance: exploit if confident, explore if uncertain
            confidence = max(h.confidence for h in active_hyps)
            score = (confidence * predicted_reward +
                     (1 - confidence) * info_gain_score)
            scored.append((score, action))

        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[0][1]

    def _predict_reward(self, action, hyps: List[Hypothesis]) -> float:
        """Weighted average predicted reward under hypothesis ensemble."""
        total_weight = sum(h.confidence for h in hyps)
        if total_weight == 0: return 0.0
        # Simplified: high-confidence hypotheses vote on action quality
        return sum(h.confidence * self._hyp_action_score(h, action)
                   for h in hyps) / total_weight

    def _hyp_action_score(self, hyp: Hypothesis, action) -> float:
        """How well does this action align with this hypothesis goal?"""
        # Domain-specific scoring — extend per environment type
        if "reach" in hyp.description and hasattr(action, 'direction'):
            return 0.6  # movement actions favored for reach goals
        return 0.3      # baseline

    def _action_info_gain(self, action, hyps: List[Hypothesis]) -> float:
        """How much info does this action give us?"""
        # Novel actions (not tried yet) have higher info gain
        if action not in self.action_history[-5:]:
            return 0.8
        return 0.2

    def update_after_step(self, action, new_obs: dict):
        """Update beliefs after receiving environment feedback."""
        reward = new_obs.get("reward", 0)
        self.action_history.append(action)

        # Falsify hypotheses inconsistent with observed reward
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
