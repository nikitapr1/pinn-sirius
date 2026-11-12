"""Tiny CPU-only training test using Python's standard library."""
import json
import os
from pathlib import Path

weight = 0.0
losses = []
for step in range(100):
    gradient = sum(2 * (weight * x - 2 * x) * x for x in (1, 2, 3)) / 3
    weight -= 0.05 * gradient
    losses.append(sum((weight * x - 2 * x) ** 2 for x in (1, 2, 3)) / 3)
assert losses[-1] < 1e-10
result = {'commit': os.environ['RUN_COMMIT'], 'steps': 100, 'weight': weight,
          'initial_loss': losses[0], 'final_loss': losses[-1]}
output = Path(os.environ['OUTPUT_DIR'])
(output / 'smoke-result.json').write_text(json.dumps(result, indent=2))
(output / 'checkpoint.json').write_text(json.dumps({'step': 100, 'weight': weight}))
print(json.dumps(result, indent=2))
