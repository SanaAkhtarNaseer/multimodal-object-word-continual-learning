import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.concept_memory import ConceptMemory


memory = ConceptMemory()

concept_a = memory.learn("shape_0")
concept_b = memory.learn("shape_1")
concept_a_again = memory.learn("shape_0")

print("Concept A ID:", concept_a)
print("Concept B ID:", concept_b)
print("Concept A again:", concept_a_again)

print("Stored concepts:")
print(memory.concepts)
