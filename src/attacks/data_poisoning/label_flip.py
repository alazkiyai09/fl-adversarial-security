"""Label-flip wrapper metadata."""


def supported_label_flip_modes() -> list[str]:
    return ["random", "targeted", "inverse"]


def load_label_flip_module():
    from src.attacks.data_poisoning.legacy.attacks.label_flip import apply_attack

    return apply_attack
