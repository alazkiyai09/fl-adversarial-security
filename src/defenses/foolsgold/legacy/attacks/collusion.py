"""
Collusion Attack Implementation.

Collusion: Multiple malicious clients coordinate to manipulate the model.
Unlike Sybil, collusion may involve more sophisticated coordination.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from flwr.common import Parameters


class CollusionAttack:
    """
    Collusion attack orchestrator.

    Multiple malicious clients coordinate their updates to manipulate
    the global model while avoiding detection.
    """

    def __init__(
        self,
        num_malicious: int,
        num_honest: int,
        collusion_strategy: str = "average_flip",
        magnitude: float = 1.0
    ):
        """
        Initialize collusion attack.

        Args:
            num_malicious: Number of colluding clients
            num_honest: Total number of clients
            collusion_strategy: Strategy for coordination
                - 'average_flip': Flip average of honest updates
                - 'scaled_flip': Scale and flip honest updates
                - 'diverse': Different malicious updates targeting same goal
            magnitude: Attack magnitude
        """
        self.num_malicious = num_malicious
        self.num_honest = num_honest
        self.collusion_strategy = collusion_strategy
        self.magnitude = magnitude

        # Malicious client IDs
        self.malicious_ids = list(range(num_honest - num_malicious, num_honest))

    def is_malicious(self, client_id: int) -> bool:
        """Check if client is malicious."""
        return client_id in self.malicious_ids

    def compute_honest_average(
        self,
        updates: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """Compute average of honest updates."""
        honest_grads = [updates[cid] for cid in updates.keys() if not self.is_malicious(cid)]

        if not honest_grads:
            return np.zeros(10)  # Placeholder

        return np.mean(honest_grads, axis=0)

    def generate_colluding_updates(
        self,
        honest_updates: Dict[int, np.ndarray],
        round_num: int = 0,
        random_seed: Optional[int] = None
    ) -> Dict[int, np.ndarray]:
        """
        Generate colluding updates.

        Args:
            honest_updates: Dictionary of honest client updates
            round_num: Current round number (for adaptive strategies)
            random_seed: Random seed

        Returns:
            Dictionary with all updates (including colluding)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        all_updates = {}

        # Copy honest updates
        for cid, grad in honest_updates.items():
            if not self.is_malicious(cid):
                all_updates[cid] = grad

        # Compute honest average
        honest_avg = self.compute_honest_average(honest_updates)

        # Generate colluding updates
        malicious_updates = []

        if self.collusion_strategy == "average_flip":
            # All colluders send the same flipped average
            malicious_base = -self.magnitude * honest_avg
            malicious_updates = [malicious_base.copy() for _ in range(self.num_malicious)]

        elif self.collusion_strategy == "scaled_flip":
            # Each colluder sends a scaled version of flipped average
            malicious_base = -self.magnitude * honest_avg
            scales = np.linspace(0.5, 1.5, self.num_malicious)
            malicious_updates = [malicious_base * s for s in scales]

        elif self.collusion_strategy == "diverse":
            # Each colluder sends a different but coordinated update
            # All push in similar general direction but with noise
            direction = -honest_avg / (np.linalg.norm(honest_avg) + 1e-8)
            for i in range(self.num_malicious):
                noise = np.random.randn(*direction.shape) * 0.1
                malicious_update = self.magnitude * (direction + noise)
                malicious_update = malicious_update / (np.linalg.norm(malicious_update) + 1e-8)
                malicious_update *= np.linalg.norm(honest_avg)
                malicious_updates.append(malicious_update)

        else:
            raise ValueError(f"Unknown collusion strategy: {self.collusion_strategy}")

        # Assign to malicious client IDs
        for i, cid in enumerate(self.malicious_ids):
            all_updates[cid] = malicious_updates[i]

        return all_updates

    def get_collusion_similarity(
        self,
        updates: Dict[int, np.ndarray]
    ) -> float:
        """
        Compute average similarity among colluding clients.

        Args:
            updates: Dictionary of client updates

        Returns:
            Average cosine similarity between colluding clients
        """
        colluder_grads = [updates[cid] for cid in self.malicious_ids if cid in updates]

        if len(colluder_grads) < 2:
            return 0.0

        # Compute pairwise similarities
        similarities = []
        for i in range(len(colluder_grads)):
            for j in range(i + 1, len(colluder_grads)):
                grad_i = colluder_grads[i]
                grad_j = colluder_grads[j]

                norm_i = np.linalg.norm(grad_i)
                norm_j = np.linalg.norm(grad_j)

                if norm_i > 0 and norm_j > 0:
                    sim = np.dot(grad_i, grad_j) / (norm_i * norm_j)
                    similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 0.0
