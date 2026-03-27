"""
FL-Specific Membership Inference Attacks

This module implements membership inference attacks specifically
designed for Federated Learning systems.

Attack Scenarios:
1. Global model attack: Infer membership in overall FL training
2. Local model attack: Infer membership in specific client's local data
3. Temporal attack: Track membership across FL rounds

Threat Model:
- Attacker can query global model
- Attacker may have access to individual client models (in some scenarios)
- Attacker can observe model evolution across FL rounds
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional
import copy


def attack_global_model(
    global_model: nn.Module,
    member_data: DataLoader,
    nonmember_data: DataLoader,
    attack_fn,
    attack_name: str,
    device: str = 'cpu',
    **attack_kwargs
) -> Dict:
    """
    Attack the global FL model to infer aggregate membership.

    This tests whether the attacker can determine if a data point
    was used in ANY client's training during FL.

    Args:
        global_model: Trained global FL model
        member_data: Data from FL training (union of all clients)
        nonmember_data: Data NOT from FL training
        attack_fn: Attack function to use (e.g., loss_based_attack)
        attack_name: Name of attack for reporting
        device: Device to run on
        **attack_kwargs: Additional arguments for attack function

    Returns:
        Dictionary with attack results
    """
    print(f"\nAttacking global model using {attack_name}...")

    # Execute attack
    all_scores, true_labels, (member_vals, nonmember_vals) = attack_fn(
        target_model=global_model,
        member_data=member_data,
        nonmember_data=nonmember_data,
        device=device,
        **attack_kwargs
    )

    return {
        'attack_name': f"global_{attack_name}",
        'scores': all_scores,
        'labels': true_labels,
        'member_values': member_vals,
        'nonmember_values': nonmember_vals
    }


def attack_local_models(
    client_models: List[nn.Module],
    client_member_data: List[DataLoader],
    client_nonmember_data: List[DataLoader],
    attack_fn,
    attack_name: str,
    device: str = 'cpu',
    **attack_kwargs
) -> List[Dict]:
    """
    Attack individual client models to infer local membership.

    This tests whether the attacker can determine if a data point
    was used by a SPECIFIC client during FL.

    Args:
        client_models: List of client models
        client_member_data: List of member data loaders for each client
        client_nonmember_data: List of non-member data loaders for each client
        attack_fn: Attack function to use
        attack_name: Name of attack for reporting
        device: Device to run on
        **attack_kwargs: Additional arguments for attack function

    Returns:
        List of attack results per client
    """
    print(f"\nAttacking {len(client_models)} local models using {attack_name}...")

    results = []

    for i, (client_model, member_loader, nonmember_loader) in enumerate(
        zip(client_models, client_member_data, client_nonmember_data)
    ):
        print(f"  Client {i+1}/{len(client_models)}...")

        all_scores, true_labels, (member_vals, nonmember_vals) = attack_fn(
            target_model=client_model,
            member_data=member_loader,
            nonmember_data=nonmember_loader,
            device=device,
            **attack_kwargs
        )

        results.append({
            'client_id': i,
            'attack_name': f"client_{i}_{attack_name}",
            'scores': all_scores,
            'labels': true_labels,
            'member_values': member_vals,
            'nonmember_values': nonmember_vals
        })

    return results


def temporal_attack(
    model_history: List[nn.Module],
    data_point: torch.Tensor,
    true_label: torch.Tensor,
    round_indices: Optional[List[int]] = None,
    attack_fn: Optional[callable] = None,
    device: str = 'cpu'
) -> Dict:
    """
    Track membership probability of a single data point across FL rounds.

    This tests how membership signal evolves during FL training.

    Args:
        model_history: List of global models from different FL rounds
        data_point: Single data point to analyze
        true_label: True label for data point
        round_indices: Specific rounds to analyze (default: all)
        attack_fn: Optional function to compute membership score
        device: Device to run on

    Returns:
        Dictionary with temporal membership scores
    """
    if round_indices is None:
        round_indices = list(range(len(model_history)))

    print(f"\nRunning temporal attack across {len(round_indices)} FL rounds...")

    data_point = data_point.unsqueeze(0).to(device)  # Add batch dimension
    true_label = torch.tensor([true_label]).to(device)

    temporal_scores = []

    for round_idx in round_indices:
        model = model_history[round_idx]
        model.eval()

        with torch.no_grad():
            logits = model(data_point)
            probs = torch.softmax(logits, dim=1)
            max_conf = probs.max(dim=1)[0].item()
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).item()

        temporal_scores.append({
            'round': round_idx,
            'max_confidence': max_conf,
            'entropy': entropy,
            'neg_entropy': -entropy
        })

    return {
        'temporal_scores': temporal_scores,
        'round_indices': round_indices
    }


def cross_round_attack(
    model_history: List[nn.Module],
    member_data: DataLoader,
    nonmember_data: DataLoader,
    target_round: int,
    attack_fn,
    attack_name: str,
    device: str = 'cpu',
    **attack_kwargs
) -> Dict:
    """
    Attack a specific FL round using models from other rounds.

    This tests whether membership information persists across rounds.

    Args:
        model_history: List of global models from different FL rounds
        member_data: Member data
        nonmember_data: Non-member data
        target_round: Round to attack
        attack_fn: Attack function to use
        attack_name: Name of attack
        device: Device to run on
        **attack_kwargs: Additional arguments

    Returns:
        Dictionary with cross-round attack results
    """
    print(f"\nCross-round attack: Attacking round {target_round}...")

    target_model = model_history[target_round]

    all_scores, true_labels, (member_vals, nonmember_vals) = attack_fn(
        target_model=target_model,
        member_data=member_data,
        nonmember_data=nonmember_data,
        device=device,
        **attack_kwargs
    )

    return {
        'target_round': target_round,
        'attack_name': f"cross_round_{target_round}_{attack_name}",
        'scores': all_scores,
        'labels': true_labels,
        'member_values': member_vals,
        'nonmember_values': nonmember_vals
    }


def analyze_vulnerability_by_round(
    model_history: List[nn.Module],
    member_data: DataLoader,
    nonmember_data: DataLoader,
    attack_fn,
    attack_name: str,
    device: str = 'cpu',
    **attack_kwargs
) -> List[Dict]:
    """
    Analyze how vulnerability evolves across FL rounds.

    Args:
        model_history: List of global models from different FL rounds
        member_data: Member data
        nonmember_data: Non-member data
        attack_fn: Attack function to use
        attack_name: Name of attack
        device: Device to run on
        **attack_kwargs: Additional arguments

    Returns:
        List of attack results per round
    """
    print(f"\nAnalyzing vulnerability across {len(model_history)} FL rounds...")

    results = []

    for round_idx, model in enumerate(model_history):
        print(f"  Round {round_idx+1}/{len(model_history)}...")

        all_scores, true_labels, _ = attack_fn(
            target_model=model,
            member_data=member_data,
            nonmember_data=nonmember_data,
            device=device,
            **attack_kwargs
        )

        # Compute AUC for this round
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(true_labels, all_scores)

        results.append({
            'round': round_idx,
            'auc': auc,
            'scores': all_scores,
            'labels': true_labels
        })

    return results


def client_data_inference_attack(
    global_model: nn.Module,
    client_models: List[nn.Module],
    test_data: DataLoader,
    device: str = 'cpu'
) -> Dict:
    """
    Infer which client's local training data a sample came from.

    This is a different attack: instead of member/non-member,
    we predict which client trained on the data.

    Args:
        global_model: Global FL model
        client_models: List of client models
        test_data: Data to classify
        device: Device to run on

    Returns:
        Dictionary with client predictions
    """
    print(f"\nClient data inference attack on {len(test_data.dataset)} samples...")

    global_model.eval()
    for model in client_models:
        model.eval()

    all_client_losses = []

    # Compute loss under global model
    criterion = nn.CrossEntropyLoss(reduction='none')
    global_losses = []

    with torch.no_grad():
        for x, y in test_data:
            x, y = x.to(device), y.to(device)
            logits = global_model(x)
            loss = criterion(logits, y)
            global_losses.extend(loss.cpu().numpy())

    # Compute loss under each client model
    for client_idx, client_model in enumerate(client_models):
        client_losses = []

        with torch.no_grad():
            for x, y in test_data:
                x, y = x.to(device), y.to(device)
                logits = client_model(x)
                loss = criterion(logits, y)
                client_losses.extend(loss.cpu().numpy())

        all_client_losses.append(client_losses)

    # Predict: sample belongs to client with lowest loss
    all_client_losses = np.array(all_client_losses)  # [n_clients, n_samples]
    predicted_clients = np.argmin(all_client_losses, axis=0)

    return {
        'predicted_clients': predicted_clients,
        'client_losses': all_client_losses,
        'global_losses': np.array(global_losses)
    }


def aggregate_fl_attacks(
    global_model: nn.Module,
    client_models: Optional[List[nn.Module]],
    model_history: Optional[List[nn.Module]],
    member_data: DataLoader,
    nonmember_data: DataLoader,
    attack_fn_list: List[Tuple[str, callable]],
    device: str = 'cpu'
) -> Dict[str, List[Dict]]:
    """
    Run comprehensive FL attack evaluation.

    Args:
        global_model: Global FL model
        client_models: Optional list of client models
        model_history: Optional list of models across rounds
        member_data: Member data
        nonmember_data: Non-member data
        attack_fn_list: List of (attack_name, attack_function) tuples
        device: Device to run on

    Returns:
        Dictionary of attack results by type
    """
    results = {
        'global_attacks': [],
        'local_attacks': [],
        'temporal_attacks': [],
        'cross_round_attacks': []
    }

    # Global model attacks
    print("\n" + "="*80)
    print("GLOBAL MODEL ATTACKS")
    print("="*80)

    for attack_name, attack_fn in attack_fn_list:
        result = attack_global_model(
            global_model=global_model,
            member_data=member_data,
            nonmember_data=nonmember_data,
            attack_fn=attack_fn,
            attack_name=attack_name,
            device=device
        )
        results['global_attacks'].append(result)

    # Local model attacks (if client models available)
    if client_models is not None:
        print("\n" + "="*80)
        print("LOCAL MODEL ATTACKS")
        print("="*80)

        # For simplicity, use same member/non-member data for all clients
        # In practice, you'd have client-specific data
        client_member_data = [member_data] * len(client_models)
        client_nonmember_data = [nonmember_data] * len(client_models)

        for attack_name, attack_fn in attack_fn_list:
            client_results = attack_local_models(
                client_models=client_models,
                client_member_data=client_member_data,
                client_nonmember_data=client_nonmember_data,
                attack_fn=attack_fn,
                attack_name=attack_name,
                device=device
            )
            results['local_attacks'].extend(client_results)

    # Temporal attacks (if model history available)
    if model_history is not None and len(model_history) > 1:
        print("\n" + "="*80)
        print("TEMPORAL ATTACKS")
        print("="*80)

        # Analyze vulnerability across rounds
        for attack_name, attack_fn in attack_fn_list:
            round_results = analyze_vulnerability_by_round(
                model_history=model_history,
                member_data=member_data,
                nonmember_data=nonmember_data,
                attack_fn=attack_fn,
                attack_name=attack_name,
                device=device
            )
            results['temporal_attacks'].append({
                'attack_name': attack_name,
                'round_results': round_results
            })

    return results


if __name__ == "__main__":
    print("This module provides FL-specific membership inference attacks.")
    print("Use via experiment scripts.")
