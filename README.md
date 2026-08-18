# Continual Multimodal Associative Memory

## MS Research Baseline Implementation

### Overview

This repository contains an MS-level research baseline developed to study and implement the fundamental mechanisms of continual multimodal learning and associative memory.

The project investigates how a learning system can incrementally construct internal representations of visual concepts, associate information originating from multiple feature channels, and retrieve previously learned knowledge without requiring conventional batch retraining.

The implementation is inspired by the principles of Conceptual Distributed Self-organizing Networks (CDSN) and recurrent-kernel-based episodic memory. Rather than treating the reference architecture as a black-box model, the baseline was implemented modularly to understand the behavior of its individual memory components and their interaction during sequential learning.

### Technical Objectives

The baseline was developed to gain practical understanding of:

- continual and incremental concept learning;
- self-organizing receptive-field formation;
- multimodal feature representation;
- concept and associative memory construction;
- visual-to-semantic and semantic-to-visual recall;
- stability-plasticity behavior during sequential learning;
- memory growth as new concepts are introduced;
- catastrophic-forgetting evaluation;
- threshold sensitivity in self-organizing memories;
- limitations of handcrafted representations under real-world visual variability.

### Baseline Architecture

The implemented pipeline follows the conceptual flow:

Image
→ Feature Extraction
→ Shape Representation (23-D)
→ Color Representation (63-D)
→ Receptive Memories
→ Concept Formation
→ Associative Memory
→ Semantic Concept / Object Name

Two complementary visual representations are extracted independently and processed through self-organizing receptive memories.

The learned receptive representations are subsequently linked through an associative-memory layer, allowing relationships between visual characteristics and semantic concepts to be formed incrementally.

### Continual Learning Protocol

The system is evaluated in a sequential learning setting rather than training all concepts simultaneously.

The experimental protocol introduces one group of concepts during an initial learning phase and subsequently introduces previously unseen concepts while preserving the existing memory state.

This allows the implementation to study:

1. acquisition of initial concepts;
2. incorporation of new concepts without model reinitialization;
3. retention of previously acquired associations;
4. evolution of receptive and associative memory structures;
5. stability-plasticity behavior;
6. ambiguity and unsuccessful recall;
7. changes in old-concept performance after new learning.

### What This Baseline Established

The primary purpose of this version is architectural and experimental validation rather than development of a state-of-the-art image classifier.

Through this implementation, the project established a complete experimental pipeline for:

- converting real images into multimodal feature representations;
- constructing self-organizing receptive memories;
- dynamically forming concept representations;
- creating multimodal associative relationships;
- performing bidirectional associative recall;
- learning new concepts sequentially without resetting previously constructed memory;
- measuring retention and forgetting;
- analyzing class separability and threshold behavior;
- evaluating the limitations of handcrafted visual representations in heterogeneous imagery.

An important observation from the baseline experiments is that maintaining learned associations and achieving discriminative visual generalization are separate problems. The memory architecture can preserve and extend its internal associative structure while the underlying handcrafted representation may still be insufficient to distinguish visually variable observations reliably.

This observation provides the technical motivation for the next stage of the work.

### Research Extension

The next version will investigate replacing or augmenting handcrafted representations with learned representations and recurrent memory.

The planned direction is:

Real Image
→ Deep Visual Feature Extraction
→ Learned Compact Representation
→ GRU / Recurrent Memory
→ Continual Concept Representation
→ Associative Retrieval
→ Retention and Forgetting Analysis

The objective is to investigate whether learned visual representations and recurrent temporal/memory mechanisms can improve generalization while retaining the incremental-learning properties explored in the baseline.

### Research Status

**Version 1: MS Research Baseline**

Implemented:

- real-image feature extraction;
- shape and color representations;
- self-organizing receptive memory;
- concept memory;
- associative memory;
- incremental learning;
- bidirectional recall;
- class-separability analysis;
- threshold analysis;
- sequential concept introduction;
- retention and forgetting evaluation.
