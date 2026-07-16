
# Decides when to fire evidence acquisition.
# Applies confidence threshold and per-attack-type cooldown
# to prevent duplicate evidence bundles for the same event.


import time

CONFIDENCE_THRESHOLD = 0.80    # minimum confidence to trigger acquisition
COOLDOWN_SECONDS     = 30      # seconds before same attack type triggers again

_last_triggered = {}           # attack_type → last trigger timestamp


def should_acquire(attack_type: str, confidence: float) -> bool:
    
    # Returns True if evidence acquisition should fire.
    # False if attack is Benign, confidence too low, or in cooldown.
    if attack_type == "Benign":
        return False

    if confidence < CONFIDENCE_THRESHOLD:
        return False

    now  = time.time()
    last = _last_triggered.get(attack_type, 0)

    if now - last < COOLDOWN_SECONDS:
        return False

    _last_triggered[attack_type] = now
    return True