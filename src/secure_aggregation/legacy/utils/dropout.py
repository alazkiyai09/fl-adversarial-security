"""
Dropout simulation utilities.

Simulate realistic client dropout patterns for testing
the secure aggregation protocol's resilience.
"""

import random
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class DropoutPattern(Enum):
    """Patterns of client dropouts."""

    RANDOM = "random"  # Each client drops independently
    CORRELATED = "correlated"  # Groups of clients drop together
    ADVERSARIAL = "adversarial"  # Specific target clients drop
    GRADUAL = "gradual"  # Dropouts occur gradually over time


@dataclass
class DropoutScenario:
    """A specific dropout scenario."""

    active_clients: List[int]
    dead_clients: List[int]
    pattern: DropoutPattern
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def dropout_rate(self) -> float:
        """Calculate dropout rate."""
        total = len(self.active_clients) + len(self.dead_clients)
        if total == 0:
            return 0.0
        return len(self.dead_clients) / total


class DropoutSimulator:
    """
    Simulate various client dropout scenarios.

    Used for testing the protocol's resilience to different
    dropout patterns and rates.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the dropout simulator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        self.seed = seed

    def simulate_random(
        self,
        client_ids: List[int],
        dropout_rate: float
    ) -> DropoutScenario:
        """
        Simulate random independent dropouts.

        Args:
            client_ids: All client IDs
            dropout_rate: Probability of each client dropping (0.0 to 1.0)

        Returns:
            DropoutScenario with active and dead clients
        """
        dead_clients = [
            cid for cid in client_ids
            if random.random() < dropout_rate
        ]
        active_clients = [cid for cid in client_ids if cid not in dead_clients]

        return DropoutScenario(
            active_clients=active_clients,
            dead_clients=dead_clients,
            pattern=DropoutPattern.RANDOM,
            metadata={'dropout_rate': dropout_rate}
        )

    def simulate_correlated(
        self,
        client_ids: List[int],
        num_groups: int = 3,
        group_dropout_prob: float = 0.3
    ) -> DropoutScenario:
        """
        Simulate correlated dropouts (groups fail together).

        Simulates scenarios where clients share infrastructure
        (e.g., same data center) and fail together.

        Args:
            client_ids: All client IDs
            num_groups: Number of groups to divide clients into
            group_dropout_prob: Probability each group drops

        Returns:
            DropoutScenario with correlated failures
        """
        # Divide clients into groups
        sorted_clients = sorted(client_ids)
        groups = []
        group_size = len(sorted_clients) // num_groups

        for i in range(num_groups):
            start = i * group_size
            end = start + group_size if i < num_groups - 1 else len(sorted_clients)
            groups.append(sorted_clients[start:end])

        # Randomly drop entire groups
        dead_clients = []
        for group in groups:
            if random.random() < group_dropout_prob:
                dead_clients.extend(group)

        active_clients = [cid for cid in client_ids if cid not in dead_clients]

        return DropoutScenario(
            active_clients=active_clients,
            dead_clients=dead_clients,
            pattern=DropoutPattern.CORRELATED,
            metadata={
                'num_groups': num_groups,
                'groups_dropped': sum(1 for g in groups if any(cid in dead_clients for cid in g))
            }
        )

    def simulate_adversarial(
        self,
        client_ids: List[int],
        target_clients: List[int]
    ) -> DropoutScenario:
        """
        Simulate adversarial dropout of specific clients.

        Args:
            client_ids: All client IDs
            target_clients: Specific clients to drop

        Returns:
            DropoutScenario with targeted dropouts
        """
        dead_clients = [cid for cid in target_clients if cid in client_ids]
        active_clients = [cid for cid in client_ids if cid not in dead_clients]

        return DropoutScenario(
            active_clients=active_clients,
            dead_clients=dead_clients,
            pattern=DropoutPattern.ADVERSARIAL,
            metadata={'target_clients': target_clients}
        )

    def simulate_gradual(
        self,
        client_ids: List[int],
        phases: int = 3,
        per_phase_rate: float = 0.1
    ) -> List[DropoutScenario]:
        """
        Simulate gradual dropouts over multiple phases.

        Args:
            client_ids: All client IDs
            phases: Number of dropout phases
            per_phase_rate: Dropout rate per phase

        Returns:
            List of DropoutScenario, one per phase
        """
        scenarios = []
        currently_dead: Set[int] = set()
        currently_active = set(client_ids)

        for phase in range(phases):
            # Some active clients drop
            active_list = list(currently_active)
            new_dropouts = [
                cid for cid in active_list
                if random.random() < per_phase_rate
            ]

            currently_dead.update(new_dropouts)
            currently_active.difference_update(new_dropouts)

            scenarios.append(DropoutScenario(
                active_clients=list(currently_active),
                dead_clients=list(currently_dead),
                pattern=DropoutPattern.GRADUAL,
                metadata={
                    'phase': phase,
                    'new_dropouts_this_phase': new_dropouts
                }
            ))

        return scenarios

    def simulate_edge_cases(
        self,
        client_ids: List[int],
        threshold: int
    ) -> List[DropoutScenario]:
        """
        Generate edge case scenarios for testing.

        Args:
            client_ids: All client IDs
            threshold: Secret sharing threshold

        Returns:
            List of edge case scenarios
        """
        scenarios = []
        n = len(client_ids)

        # Edge case 1: Exactly threshold clients (minimum viable)
        scenarios.append(DropoutScenario(
            active_clients=client_ids[:threshold],
            dead_clients=client_ids[threshold:],
            pattern=DropoutPattern.RANDOM,
            metadata={'description': 'Exactly threshold clients active'}
        ))

        # Edge case 2: Threshold - 1 clients (should fail)
        if threshold > 1:
            scenarios.append(DropoutScenario(
                active_clients=client_ids[:threshold - 1],
                dead_clients=client_ids[threshold - 1:],
                pattern=DropoutPattern.RANDOM,
                metadata={'description': 'Below threshold - should fail'}
            ))

        # Edge case 3: All clients active
        scenarios.append(DropoutScenario(
            active_clients=client_ids[:],
            dead_clients=[],
            pattern=DropoutPattern.RANDOM,
            metadata={'description': 'All clients active'}
        ))

        # Edge case 4: 30% dropout (common test case)
        num_dead = int(n * 0.3)
        scenarios.append(DropoutScenario(
            active_clients=client_ids[:n - num_dead],
            dead_clients=client_ids[n - num_dead:],
            pattern=DropoutPattern.RANDOM,
            metadata={'description': '30% dropout'}
        ))

        return scenarios


def analyze_scenario(
    scenario: DropoutScenario,
    threshold: int
) -> dict:
    """
    Analyze whether a scenario is recoverable.

    Args:
        scenario: Dropout scenario to analyze
        threshold: Secret sharing threshold

    Returns:
        Dictionary with analysis results
    """
    num_active = len(scenario.active_clients)
    num_dead = len(scenario.dead_clients)

    can_recover = num_active >= threshold

    return {
        'num_active': num_active,
        'num_dead': num_dead,
        'dropout_rate': scenario.dropout_rate,
        'threshold': threshold,
        'can_recover': can_recover,
        'margin': num_active - threshold,
        'pattern': scenario.pattern.value
    }


# Convenience function matching the import in dropout_recovery.py
def simulate_dropouts(
    client_ids: List[int],
    dropout_rate: float,
    seed: Optional[int] = None
) -> Tuple[List[int], List[int]]:
    """
    Simulate random client dropouts.

    Args:
        client_ids: List of all client IDs
        dropout_rate: Probability of each client dropping out (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (active_client_ids, dead_client_ids)
    """
    simulator = DropoutSimulator(seed=seed)
    scenario = simulator.simulate_random(client_ids, dropout_rate)

    return scenario.active_clients, scenario.dead_clients
