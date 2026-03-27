"""
Attack simulation engine for FL security demonstrations.
Supports label flipping, backdoor, Byzantine, and poisoning attacks.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from .data_models import (
    AttackConfig,
    ClientMetric,
    TrainingRound,
    SecurityEvent
)


class AttackEngine:
    """
    Simulates various attacks on federated learning.
    Designed for demonstration and educational purposes.
    """

    def __init__(self, config: AttackConfig, seed: int = 42):
        """
        Initialize attack engine.

        Args:
            config: Attack configuration
            seed: Random seed for reproducibility
        """
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.attack_active = False
        self.attack_effects_log: List[Dict] = []

    def should_attack_this_round(self, round_num: int) -> bool:
        """
        Determine if attack should be active in current round.

        Args:
            round_num: Current training round

        Returns:
            True if attack is active this round
        """
        if round_num < self.config.start_round:
            return False

        if self.config.end_round is not None and round_num > self.config.end_round:
            return False

        return True

    def apply_attack(
        self,
        round_num: int,
        client_updates: Dict[int, np.ndarray],
        client_metrics: Dict[int, ClientMetric]
    ) -> Tuple[Dict[int, np.ndarray], List[SecurityEvent]]:
        """
        Apply attack to client updates.

        Args:
            round_num: Current round number
            client_updates: Dictionary mapping client_id to update vector
            client_metrics: Dictionary mapping client_id to client metrics

        Returns:
            Tuple of (modified_updates, security_events)
        """
        events = []

        if not self.should_attack_this_round(round_num):
            return client_updates, events

        self.attack_active = True

        # Determine which clients are attackers
        attacker_ids = self._get_attacker_ids(list(client_updates.keys()))

        # Apply attack-specific logic
        if self.config.attack_type == "label_flipping":
            modified_updates, new_events = self._apply_label_flipping(
                round_num, client_updates, attacker_ids
            )
        elif self.config.attack_type == "backdoor":
            modified_updates, new_events = self._apply_backdoor(
                round_num, client_updates, attacker_ids
            )
        elif self.config.attack_type == "byzantine":
            modified_updates, new_events = self._apply_byzantine(
                round_num, client_updates, attacker_ids
            )
        elif self.config.attack_type == "poisoning":
            modified_updates, new_events = self._apply_poisoning(
                round_num, client_updates, attacker_ids
            )
        else:
            modified_updates, new_events = client_updates, []

        events.extend(new_events)
        self.attack_effects_log.append({
            "round": round_num,
            "attackers": attacker_ids,
            "effects": len(new_events)
        })

        return modified_updates, events

    def _get_attacker_ids(self, all_client_ids: List[int]) -> List[int]:
        """Determine which client IDs are attackers."""
        if self.config.attacker_ids:
            # Use specified attacker IDs
            return [cid for cid in self.config.attacker_ids if cid in all_client_ids]
        else:
            # Randomly select attackers
            n_attackers = min(self.config.num_attackers, len(all_client_ids))
            return sorted(self.rng.choice(all_client_ids, n_attackers, replace=False))

    def _apply_label_flipping(
        self,
        round_num: int,
        client_updates: Dict[int, np.ndarray],
        attacker_ids: List[int]
    ) -> Tuple[Dict[int, np.ndarray], List[SecurityEvent]]:
        """
        Apply label flipping attack.
        Attackers flip gradient signs to push model in wrong direction.
        """
        events = []
        modified_updates = client_updates.copy()

        for attacker_id in attacker_ids:
            if attacker_id not in modified_updates:
                continue

            # Flip signs of gradients (label flipping effect)
            update = modified_updates[attacker_id]
            flipped_update = -update * self.config.label_flip_ratio
            modified_updates[attacker_id] = flipped_update

            events.append(SecurityEvent(
                event_id=f"lf_{round_num}_{attacker_id}",
                event_type="attack_detected" if self.rng.rand() > 0.7 else "attack_successful",
                severity="high",
                message=f"Label flipping attack detected from Client {attacker_id}",
                round_num=round_num,
                attack_type="label_flipping",
                affected_clients=[attacker_id],
                confidence=0.85 + self.rng.rand() * 0.15
            ))

        return modified_updates, events

    def _apply_backdoor(
        self,
        round_num: int,
        client_updates: Dict[int, np.ndarray],
        attacker_ids: List[int]
    ) -> Tuple[Dict[int, np.ndarray], List[SecurityEvent]]:
        """
        Apply backdoor attack.
        Attackers inject specific pattern into gradients.
        """
        events = []
        modified_updates = client_updates.copy()

        # Create backdoor trigger pattern
        trigger_value = self.config.backdoor_trigger_pattern if self.config.backdoor_trigger_pattern > 0 else 5.0

        for attacker_id in attacker_ids:
            if attacker_id not in modified_updates:
                continue

            # Inject backdoor pattern into specific gradient positions
            update = modified_updates[attacker_id]
            backdoored_update = update.copy()

            # Add pattern to a subset of gradient indices
            n_indices = max(1, len(update) // 10)
            indices = self.rng.choice(len(update), n_indices, replace=False)
            backdoored_update[indices] += trigger_value

            modified_updates[attacker_id] = backdoored_update

            if self.rng.rand() > 0.8:  # Only detect sometimes
                events.append(SecurityEvent(
                    event_id=f"bd_{round_num}_{attacker_id}",
                    event_type="anomaly_detected",
                    severity="medium",
                    message=f"Anomalous gradient pattern detected from Client {attacker_id}",
                    round_num=round_num,
                    attack_type="backdoor",
                    affected_clients=[attacker_id],
                    confidence=0.6 + self.rng.rand() * 0.3
                ))

        return modified_updates, events

    def _apply_byzantine(
        self,
        round_num: int,
        client_updates: Dict[int, np.ndarray],
        attacker_ids: List[int]
    ) -> Tuple[Dict[int, np.ndarray], List[SecurityEvent]]:
        """
        Apply Byzantine attack.
        Attackers send malicious updates to disrupt convergence.
        """
        events = []
        modified_updates = client_updates.copy()

        for attacker_id in attacker_ids:
            if attacker_id not in modified_updates:
                continue

            update = modified_updates[attacker_id]

            if self.config.byzantine_type == "sign_flip":
                malicious_update = -update
            elif self.config.byzantine_type == "random":
                malicious_update = self.rng.randn(*update.shape) * np.std(update)
            else:  # scaled
                malicious_update = update * self.config.poison_magnitude

            modified_updates[attacker_id] = malicious_update

            events.append(SecurityEvent(
                event_id=f"bz_{round_num}_{attacker_id}",
                event_type="attack_detected",
                severity="high",
                message=f"Byzantine attack ({self.config.byzantine_type}) from Client {attacker_id}",
                round_num=round_num,
                attack_type="byzantine",
                affected_clients=[attacker_id],
                confidence=0.75 + self.rng.rand() * 0.25
            ))

        return modified_updates, events

    def _apply_poisoning(
        self,
        round_num: int,
        client_updates: Dict[int, np.ndarray],
        attacker_ids: List[int]
    ) -> Tuple[Dict[int, np.ndarray], List[SecurityEvent]]:
        """
        Apply data/model poisoning attack.
        Similar to Byzantine but with different characteristics.
        """
        events = []
        modified_updates = client_updates.copy()

        for attacker_id in attacker_ids:
            if attacker_id not in modified_updates:
                continue

            # Apply strong poisoning by scaling gradients
            update = modified_updates[attacker_id]
            poisoned_update = update * self.config.poison_magnitude

            # Add random noise to hide attack
            noise = self.rng.randn(*update.shape) * 0.1 * np.std(update)
            poisoned_update += noise

            modified_updates[attacker_id] = poisoned_update

            events.append(SecurityEvent(
                event_id=f"ps_{round_num}_{attacker_id}",
                event_type="attack_detected",
                severity="critical",
                message=f"Model poisoning detected from Client {attacker_id}",
                round_num=round_num,
                attack_type="poisoning",
                affected_clients=[attacker_id],
                confidence=0.9 + self.rng.rand() * 0.1
            ))

        return modified_updates, events

    def get_attack_statistics(self) -> Dict:
        """Get statistics about attack effects."""
        return {
            "attack_type": self.config.attack_type,
            "total_rounds_active": len(self.attack_effects_log),
            "total_effects": sum(log["effects"] for log in self.attack_effects_log),
            "effects_by_round": self.attack_effects_log
        }
