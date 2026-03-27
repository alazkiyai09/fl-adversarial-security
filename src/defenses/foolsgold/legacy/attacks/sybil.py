"""
Sybil Attack Implementation.

Sybil attack: Multiple fake clients (Sybils) send coordinated malicious updates.
Key characteristic: Sybils have highly similar gradients to each other.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from flwr.common import Parameters, ndarrays_to_parameters


def generate_sybil_updates(
    malicious_update: np.ndarray,
    num_sybils: int,
    noise_level: float = 0.0,
    random_seed: Optional[int] = None
) -> List[np.ndarray]:
    """
    Generate Sybil updates (coordinated malicious clients).

    Args:
        malicious_update: Base malicious gradient vector
        num_sybils: Number of fake Sybil clients
        noise_level: Small noise to appear distinct (default: 0.0 = identical)
        random_seed: Random seed for reproducibility

    Returns:
        List of malicious updates for Sybil clients
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    sybil_updates = []

    for _ in range(num_sybils):
        if noise_level > 0:
            # Add small noise to appear slightly different
            noise = np.random.randn(*malicious_update.shape) * noise_level
            sybil_update = malicious_update + noise
        else:
            # Identical updates (perfectly coordinated)
            sybil_update = malicious_update.copy()

        sybil_updates.append(sybil_update)

    return sybil_updates


def create_malicious_update(
    honest_update: np.ndarray,
    attack_type: str = "sign_flip",
    magnitude: float = 1.0
) -> np.ndarray:
    """
    Create malicious update from honest update.

    Args:
        honest_update: Original honest gradient
        attack_type: Type of attack ('sign_flip', 'magnitude', 'direction')
        magnitude: Attack magnitude

    Returns:
        Malicious gradient vector
    """
    if attack_type == "sign_flip":
        # Flip the sign (opposite direction)
        return -magnitude * honest_update

    elif attack_type == "magnitude":
        # Amplify the gradient
        return magnitude * honest_update

    elif attack_type == "direction":
        # Push in specific malicious direction
        # Create update that maximizes loss on a target
        malicious_dir = np.random.randn(*honest_update.shape)
        malicious_dir = malicious_dir / np.linalg.norm(malicious_dir)
        return magnitude * malicious_dir * np.linalg.norm(honest_update)

    else:
        raise ValueError(f"Unknown attack type: {attack_type}")


class SybilAttack:
    """
    Sybil attack orchestrator.

    Manages multiple malicious client IDs and generates
    coordinated updates for them.
    """

    def __init__(
        self,
        num_malicious: int,
        num_honest: int,
        attack_type: str = "sign_flip",
        noise_level: float = 0.0,
        magnitude: float = 1.0
    ):
        """
        Initialize Sybil attack.

        Args:
            num_malicious: Number of Sybil clients
            num_honest: Total number of clients (including malicious)
            attack_type: Type of malicious update
            noise_level: Noise level to appear distinct
            magnitude: Attack magnitude
        """
        self.num_malicious = num_malicious
        self.num_honest = num_honest
        self.attack_type = attack_type
        self.noise_level = noise_level
        self.magnitude = magnitude

        # Malicious client IDs (last num_malicious clients)
        self.malicious_ids = list(range(num_honest - num_malicious, num_honest))

    def is_malicious(self, client_id: int) -> bool:
        """Check if client is malicious."""
        return client_id in self.malicious_ids

    def generate_updates(
        self,
        honest_updates: Dict[int, np.ndarray],
        random_seed: Optional[int] = None
    ) -> Dict[int, np.ndarray]:
        """
        Generate updates for all clients (honest + malicious).

        Args:
            honest_updates: Dictionary {client_id: honest_gradient}
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary with all updates (including Sybils)
        """
        all_updates = {}

        # Copy honest updates
        for cid, grad in honest_updates.items():
            if not self.is_malicious(cid):
                all_updates[cid] = grad

        # Generate malicious base update from first honest client
        if honest_updates:
            first_honest_grad = next(iter(honest_updates.values()))
            malicious_base = create_malicious_update(
                first_honest_grad,
                self.attack_type,
                self.magnitude
            )

            # Generate Sybil updates
            sybil_updates = generate_sybil_updates(
                malicious_base,
                self.num_malicious,
                self.noise_level,
                random_seed
            )

            # Assign to malicious client IDs
            for i, cid in enumerate(self.malicious_ids):
                all_updates[cid] = sybil_updates[i]

        return all_updates

    def get_sybil_similarity(
        self,
        updates: Dict[int, np.ndarray]
    ) -> float:
        """
        Compute average similarity among Sybil clients.

        Args:
            updates: Dictionary of client updates

        Returns:
            Average cosine similarity between Sybil clients
        """
        sybil_grads = [updates[cid] for cid in self.malicious_ids if cid in updates]

        if len(sybil_grads) < 2:
            return 0.0

        # Compute pairwise similarities
        similarities = []
        for i in range(len(sybil_grads)):
            for j in range(i + 1, len(sybil_grads)):
                grad_i = sybil_grads[i]
                grad_j = sybil_grads[j]

                norm_i = np.linalg.norm(grad_i)
                norm_j = np.linalg.norm(grad_j)

                if norm_i > 0 and norm_j > 0:
                    sim = np.dot(grad_i, grad_j) / (norm_i * norm_j)
                    similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 0.0
