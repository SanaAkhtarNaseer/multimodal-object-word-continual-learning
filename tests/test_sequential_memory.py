import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from self_organizing_memory import SelfOrganizingMemory


memory = SelfOrganizingMemory(
    sigma=1.0,
    threshold=0.30,
    learning_rate=0.1
)


samples = [
    [1.0, 2.0, 3.0],
    [1.1, 2.1, 3.0],
    [0.9, 1.9, 3.1],

    [5.0, 5.0, 5.0],
    [5.1, 4.9, 5.0],

    [10.0, 10.0, 10.0]
]


for i, sample in enumerate(samples):

    result = memory.learn(
        np.array(sample)
    )

    print(f"\nSample {i + 1}: {sample}")
    print("Action:", result["action"])
    print("Winner:", result["winner"])
    print("Distance:", result["distance"])
    print("Number of nodes:", len(memory.nodes))


print("\nFinal learned nodes:")

for i, node in enumerate(memory.nodes):
    print(f"Node {i}: {node}")
    