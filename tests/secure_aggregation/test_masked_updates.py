"""
Tests for masked update operations.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aggregation.masked_update import (
    apply_mask,
    cancel_mask,
    verify_mask_cancellation,
    verify_mask_cancellation_dict,
    verify_mask_security
)


class TestMaskedUpdates:
    """Test suite for masked update operations."""

    def test_apply_and_cancel_mask(self):
        """Test applying and canceling a mask."""
        update = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([0.5, 0.5, 0.5, 0.5])

        # Apply mask
        masked = apply_mask(update, mask)
        expected = torch.tensor([1.5, 2.5, 3.5, 4.5])
        assert torch.allclose(masked, expected)

        # Cancel mask
        unmasked = cancel_mask(masked, mask)
        assert torch.allclose(unmasked, update)

    def test_verify_mask_cancellation(self):
        """Test mask cancellation verification."""
        # Create masks that sum to zero
        mask1 = torch.tensor([1.0, 2.0, 3.0])
        mask2 = torch.tensor([-1.0, -2.0, -3.0])
        mask3 = torch.tensor([0.0, 0.0, 0.0])

        # These should cancel to zero
        assert verify_mask_cancellation([mask1, mask2, mask3])

        # These should not cancel
        mask4 = torch.tensor([1.0, 1.0, 1.0])
        assert not verify_mask_cancellation([mask1, mask4])

    def test_verify_mask_cancellation_dict(self):
        """Test dict-based mask cancellation verification."""
        masks = {
            0: torch.tensor([1.0, 2.0, 3.0]),
            1: torch.tensor([-1.0, -2.0, -3.0])
        }

        assert verify_mask_cancellation_dict(masks)

    def test_mask_security_verification(self):
        """Test mask security properties."""
        update = torch.randn(100)
        mask = torch.randn(100)

        security = verify_mask_security(mask, update)

        # Mask should be non-zero
        assert security['mask_nonzero']

        # Masked update should differ from original
        assert security['masked_different']

        # Shapes should match
        assert security['shapes_match']

    def test_shape_mismatch_error(self):
        """Test that shape mismatch raises error."""
        update = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([1.0, 2.0])  # Different shape

        with pytest.raises(ValueError):
            apply_mask(update, mask)

    def test_dtype_mismatch_error(self):
        """Test that dtype mismatch raises error."""
        update = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        mask = torch.tensor([1, 2, 3], dtype=torch.int32)

        with pytest.raises(ValueError):
            apply_mask(update, mask)

    def test_numerical_precision(self):
        """Test cancellation with floating point precision."""
        update = torch.randn(1000)

        # Create masks with pairwise cancellation
        masks = []
        for i in range(10):
            mask = torch.randn(1000)
            masks.append(mask)
            masks.append(-mask)  # Pair that cancels

        # Sum should be very close to zero
        assert verify_mask_cancellation(masks)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
