import os
import sys
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.multimodal_memory import MultimodalMemory


memory = MultimodalMemory(
    modalities=["visual", "audio"],
    sigma=1.0,
    threshold=0.3,
    maintenance_interval=10
)

visual_1 = np.array([1.0, 2.0, 3.0])
visual_2 = np.array([1.1, 2.1, 3.0])

audio_1 = np.array([5.0, 5.0, 5.0])

result_visual_1 = memory.learn("visual", visual_1)
result_visual_2 = memory.learn("visual", visual_2)
result_audio_1 = memory.learn("audio", audio_1)

print("\nVisual result 1:")
print(result_visual_1)

print("\nVisual result 2:")
print(result_visual_2)

print("\nAudio result:")
print(result_audio_1)

print("\nVisual concepts:")
print(memory.concept_memories["visual"].concepts)

print("\nAudio concepts:")
print(memory.concept_memories["audio"].concepts)