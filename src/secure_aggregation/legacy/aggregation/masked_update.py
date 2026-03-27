"""
Mask operations for secure model updates.

Clients add cryptographic masks to their updates before sending.
After receiving all updates, masks cancel out (sum to zero).
"""

import torch
from typing import List, Dict
import copy


def apply_mask(update: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply a cryptographic mask to a model update.

    masked_update = update + mask

    Args:
        update: Original model update tensor
        mask: Random mask tensor (same shape as update)

    Returns:
        Masked update tensor

    Raises:
        ValueError: If shapes don't match
    """
    if update.shape != mask.shape:
        raise ValueError(f"Shape mismatch: update {update.shape}, mask {mask.shape}")

    if update.dtype != mask.dtype:
        raise ValueError(f"Dtype mismatch: update {update.dtype}, mask {mask.dtype}")

    return update + mask


def cancel_mask(masked_update: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Remove a mask from a masked update.

    original_update = masked_update - mask

    Args:
        masked_update: The masked update tensor
        mask: The mask to remove

    Returns:
        Unmasked original update

    Raises:
        ValueError: If shapes don't match
    """
    if masked_update.shape != mask.shape:
        raise ValueError(f"Shape mismatch: masked_update {masked_update.shape}, mask {mask.shape}")

    if masked_update.dtype != mask.dtype:
        raise ValueError(f"Dtype mismatch: masked_update {masked_update.dtype}, mask {mask.dtype}")

    return masked_update - mask


def verify_mask_cancellation(all_masks: List[torch.Tensor]) -> bool:
    """
    Verify that all masks sum to zero (perfect cancellation).

    In secure aggregation, each client's mask should have a pair
    such that the total sum of all masks equals zero.

    Args:
        all_masks: List of all client masks

    Returns:
        True if masks sum to zero (within numerical precision)
    """
    if not all_masks:
        return True  # No masks = zero sum

    # Sum all masks
    total_mask = torch.zeros_like(all_masks[0])
    for mask in all_masks:
        total_mask = total_mask + mask

    # Check if sum is close to zero (accounting for floating point errors)
    tolerance = 1e-6
    is_zero = torch.all(torch.abs(total_mask) < tolerance)

    return is_zero.item()


def verify_mask_cancellation_dict(client_masks: Dict[int, torch.Tensor]) -> bool:
    """
    Verify mask cancellation given a dict of client_id -> mask.

    Args:
        client_masks: Dictionary mapping client IDs to their masks

    Returns:
        True if masks sum to zero
    """
    masks = list(client_masks.values())
    return verify_mask_cancellation(masks)


def compute_mask_contribution(
    my_mask: torch.Tensor,
    pairwise_masks: Dict[int, torch.Tensor]
) -> torch.Tensor:
    """
    Compute this client's contribution to the mask sum.

    Each client creates one "positive" mask for itself and contributes
    to other clients' masks via pairwise secrets.

    Args:
        my_mask: This client's own mask
        pairwise_masks: Dict of masks created with other clients

    Returns:
        Total mask contribution (should have pair with another client)
    """
    total = torch.zeros_like(my_mask)
    total = total + my_mask

    for mask in pairwise_masks.values():
        total = total + mask

    return total


def verify_mask_security(mask: torch.Tensor, update: torch.Tensor) -> Dict[str, bool]:
    """
    Verify that mask properly hides the update.

    Args:
        mask: The mask tensor
        update: The original update tensor

    Returns:
        Dictionary with security metrics
    """
    masked = apply_mask(update, mask)

    results = {
        'mask_nonzero': torch.any(mask != 0).item(),
        'masked_different': torch.any(masked != update).item(),
        'mask_magnitude_sufficient': torch.norm(mask).item() > 0.1 * torch.norm(update).item(),
        'shapes_match': mask.shape == update.shape
    }

    return results
