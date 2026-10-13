# ART Neural Network Implementations

## Description

This project demonstrates three Adaptive Resonance Theory (ART) neural network models: ART1, ART2, and Fuzzy ART, using Python and NumPy. These models are associated with unsupervised learning and pattern clustering.

The project explains how input patterns are compared with existing cluster prototypes and assigned to clusters according to matching criteria. It also demonstrates the role of the vigilance parameter in determining whether an input pattern belongs to an existing cluster or should form a new cluster.

## Objectives

- Understand the basic concepts of Adaptive Resonance Theory.
- Implement simple clustering methods using Python.
- Explore the differences between binary and continuous input patterns.
- Understand the role of vigilance and cluster prototypes.
- Compare ART1, ART2-inspired clustering, and Fuzzy ART.
- Observe how input patterns are assigned to clusters step by step.

## 1. ART1 (Adaptive Resonance Theory 1)

### Introduction

ART1 is a neural network model designed for clustering binary input patterns. Binary patterns contain only two values: 0 and 1.

Example input patterns:

- P1 = `[1, 1, 0, 0]`
- P2 = `[1, 1, 0, 1]`
- P3 = `[0, 0, 1, 1]`
- P4 = `[0, 0, 1, 0]`

ART1 groups patterns according to their similarity while allowing new clusters to be created when patterns do not sufficiently match existing categories.

### Working Principle

1. Take a binary input pattern.
2. If no cluster exists, create a new cluster.
3. Compare the input with existing cluster prototypes.
4. Calculate the match between the input and a prototype.
5. Compare the match with the vigilance parameter.
6. If the match satisfies the vigilance requirement, assign the input to the cluster and update its prototype.
7. Otherwise, check another cluster or create a new cluster if none matches.

### Important Parameters

- **Input pattern:** Binary data provided to the algorithm.
- **Prototype:** The representative pattern stored for a cluster.
- **Vigilance parameter:** Controls the minimum match required for cluster membership.
- **Cluster:** A group of patterns that satisfy the matching criterion.

### Example: Step-by-Step Clustering

The vigilance parameter is set to `0.7`.

**Step 1: Process P1 = `[1, 1, 0, 0]`**

Initially, no clusters exist. Therefore, P1 creates Cluster 1.

Cluster 1 prototype: `[1, 1, 0, 0]`

**Step 2: Process P2 = `[1, 1, 0, 1]`**

Compare P2 with Cluster 1.

The intersection contains two common active bits.

Number of active bits in P2 = 3.

Match = 2 / 3 = 0.667.

Since `0.667 < 0.7`, P2 does not join Cluster 1. It creates Cluster 2.

Cluster 2 prototype: `[1, 1, 0, 1]`

**Step 3: Process P3 = `[0, 0, 1, 1]`**

Compare P3 with Clusters 1 and 2.

- Match with Cluster 1 = 0.
- Match with Cluster 2 = 1 / 2 = 0.5.

Both values are below the vigilance threshold of 0.7. Therefore, P3 creates Cluster 3.

Cluster 3 prototype: `[0, 0, 1, 1]`

**Step 4: Process P4 = `[0, 0, 1, 0]`**

Compare P4 with the existing clusters.

- Match with Cluster 1 = 0.
- Match with Cluster 2 = 0.
- Match with Cluster 3 = 1 / 1 = 1.0.

Since `1.0 >= 0.7`, P4 joins Cluster 3.

The prototype is updated using the binary intersection.

### Final Result

- **Cluster 1:** P1 = `[1, 1, 0, 0]`
- **Cluster 2:** P2 = `[1, 1, 0, 1]`
- **Cluster 3:** P3 = `[0, 0, 1, 1]`, P4 = `[0, 0, 1, 0]`

### Applications

- Binary pattern recognition
- Grouping binary feature vectors
- Simple unsupervised classification demonstrations

## 2. ART2 (Adaptive Resonance Theory 2)

### Introduction

ART2 is designed to process continuous or analog-valued input patterns. Unlike ART1, which works with binary inputs, ART2 can process numerical values such as 0.2, 0.6, and 0.9.

Example input patterns:

- P1 = `[0.9, 0.8]`
- P2 = `[0.8, 0.9]`
- P3 = `[0.1, 0.2]`
- P4 = `[0.2, 0.1]`

The objective is to group input patterns according to their characteristics while allowing new categories to form when necessary.

### Working Principle

1. Provide a continuous-valued input pattern.
2. Check whether cluster prototypes are available.
3. Compare the input with existing prototypes.
4. Calculate similarity using the cosine similarity method used in this demonstration.
5. Compare the similarity with the vigilance threshold.
6. If the threshold is satisfied, assign the pattern to the cluster and update its prototype.
7. If no existing cluster satisfies the threshold, create a new cluster.

### Important Parameters

- **Input pattern:** A vector containing continuous numerical values.
- **Prototype:** A representative vector for an existing cluster.
- **Cosine similarity:** Measures directional similarity between two vectors in this demonstration.
- **Vigilance parameter:** Defines the minimum similarity required for assigning a pattern to a cluster.

### Example: Step-by-Step Clustering

The vigilance parameter is set to `0.5`.

The simplified program calculates cosine similarity using the following formula:

Similarity = (A · B) / (||A|| × ||B||)

Here, A is the input pattern and B is the cluster prototype.

**Step 1: Process P1 = `[0.9, 0.8]`**

Initially, no clusters exist. Therefore, P1 creates Cluster 1.

Cluster 1 prototype: `[0.9, 0.8]`

**Step 2: Process P2 = `[0.8, 0.9]`**

Compare P2 with Cluster 1.

Cosine similarity is approximately `0.993`.

Since `0.993 >= 0.5`, P2 joins Cluster 1.

The prototype is updated by averaging the two patterns:

`([0.9, 0.8] + [0.8, 0.9]) / 2 = [0.85, 0.85]`

Updated prototype: `[0.85, 0.85]`

**Step 3: Process P3 = `[0.1, 0.2]`**

Compare P3 with the updated prototype `[0.85, 0.85]`.

Cosine similarity is approximately `0.949`.

Since `0.949 >= 0.5`, P3 joins Cluster 1.

The prototype is updated again using the average of the previous prototype and P3.

Updated prototype: `[0.475, 0.525]`

**Step 4: Process P4 = `[0.2, 0.1]`**

Compare P4 with the updated prototype `[0.475, 0.525]`.

Cosine similarity is approximately `0.949`.

Since `0.949 >= 0.5`, P4 joins Cluster 1.

### Final Result

- **Cluster 1:** P1, P2, P3, P4

All four patterns belong to Cluster 1 in this simplified implementation.

This result illustrates an important limitation: cosine similarity measures direction rather than magnitude, so patterns with substantially different magnitudes may still be grouped together.

### Applications

- Continuous-pattern clustering demonstrations
- Analog feature-vector analysis
- Introductory studies of adaptive neural networks

## 3. Fuzzy ART (Fuzzy Adaptive Resonance Theory)

### Introduction

Fuzzy ART uses fuzzy set operations for pattern matching and clustering. It generally accepts continuous-valued inputs in the range 0 to 1.

Unlike a simple binary comparison, Fuzzy ART uses fuzzy intersection to compare an input pattern with an existing cluster prototype.

Example input patterns:

- P1 = `[0.9, 0.8]`
- P2 = `[0.8, 0.9]`
- P3 = `[0.1, 0.2]`
- P4 = `[0.2, 0.1]`

### Working Principle

1. Provide a continuous-valued input pattern.
2. If no cluster exists, create a new cluster.
3. Compare the input with existing cluster prototypes using fuzzy intersection.
4. Calculate the match between the input and the prototype.
5. Compare the match with the vigilance parameter.
6. If the match satisfies the threshold, assign the input to the cluster and update the prototype using fuzzy intersection.
7. If the pattern does not match any existing cluster, create a new cluster.

### Important Parameters

- **Input pattern:** A vector containing continuous values, usually between 0 and 1.
- **Prototype:** The representative vector of a cluster.
- **Fuzzy intersection:** Obtained by taking the minimum of corresponding input and prototype values.
- **Choice parameter (alpha):** A small positive parameter used in the category choice function.
- **Vigilance parameter:** Determines the minimum match required for cluster membership.

### Example: Step-by-Step Clustering

The vigilance parameter is set to `0.7`, and the choice parameter alpha is set to `0.001`.

**Step 1: Process P1 = `[0.9, 0.8]`**

Initially, no clusters exist. Therefore, P1 creates Cluster 1.

Cluster 1 prototype: `[0.9, 0.8]`

**Step 2: Process P2 = `[0.8, 0.9]`**

Compare P2 with Cluster 1.

Fuzzy intersection:

`[min(0.8, 0.9), min(0.9, 0.8)] = [0.8, 0.8]`

Sum of intersection = `1.6`.

Sum of input = `1.7`.

Match = `1.6 / 1.7 = 0.941`.

Since `0.941 >= 0.7`, P2 joins Cluster 1.

The prototype is updated using the fuzzy intersection.

Updated prototype: `[0.8, 0.8]`

**Step 3: Process P3 = `[0.1, 0.2]`**

Compare P3 with the updated prototype `[0.8, 0.8]`.

Fuzzy intersection:

`[min(0.1, 0.8), min(0.2, 0.8)] = [0.1, 0.2]`

Sum of intersection = `0.3`.

Sum of input = `0.3`.

Match = `0.3 / 0.3 = 1.0`.

Since `1.0 >= 0.7`, P3 joins Cluster 1.

The prototype is updated to `[0.1, 0.2]`.

**Step 4: Process P4 = `[0.2, 0.1]`**

Compare P4 with the updated prototype `[0.1, 0.2]`.

Fuzzy intersection:

`[min(0.2, 0.1), min(0.1, 0.2)] = [0.1, 0.1]`

Sum of intersection = `0.2`.

Sum of input = `0.3`.

Match = `0.2 / 0.3 = 0.667`.

Since `0.667 < 0.7`, P4 does not join Cluster 1. It creates Cluster 2.

### Final Result

- **Cluster 1:** P1, P2, P3
- **Cluster 2:** P4

### Applications

- Fuzzy pattern clustering
- Continuous feature-vector grouping
- Pattern recognition
- Unsupervised learning demonstrations

## Comparison of ART1, ART2, and Fuzzy ART

| Feature | ART1 | ART2 | Fuzzy ART |
|---|---|---|---|
| Input type | Binary | Continuous | Continuous, usually 0–1 |
| Comparison method in this project | Binary pattern matching | Cosine similarity | Fuzzy intersection |
| Vigilance parameter | Yes | Yes | Yes |
| Prototype update in this project | Binary intersection | Averaging | Fuzzy intersection |
| Main purpose | Binary pattern clustering | Continuous-pattern clustering | Fuzzy pattern clustering |

## Technologies Used

- Python
- NumPy

## Installation

Install Python and NumPy.

Install the required dependency using:

```bash
pip install -r requirements.txt
```

## How to Run

Run each program separately from the project directory:

```bash
python art1.py
python art2.py
python fuzzy_art.py
```

## Project Structure

- `art1.py` – Simplified ART1 binary clustering demonstration.
- `art2.py` – ART2-inspired continuous clustering demonstration.
- `fuzzy_art.py` – Simplified Fuzzy ART clustering demonstration.
- `requirements.txt` – Lists the required Python dependency.
- `README.md` – Project description and documentation.

## Note

The implementations in this repository are simplified educational demonstrations of ART1, ART2, and Fuzzy ART.

The ART1 program uses a simplified binary matching approach. The ART2 program uses cosine similarity and prototype averaging. The Fuzzy ART program demonstrates fuzzy intersection and vigilance-based matching.

These implementations do not include all the mechanisms of the complete textbook algorithms. In particular, the ART2 implementation does not include the complete ART2 network architecture, and the Fuzzy ART implementation does not use its calculated choice score for category selection. Therefore, the programs should be treated as introductory demonstrations rather than complete textbook implementations.

## Conclusion

This project introduces three ART-family approaches to clustering. ART1 focuses on binary inputs, ART2 is designed for continuous inputs, and Fuzzy ART uses fuzzy set operations for pattern matching.

The implementations demonstrate cluster formation, prototype comparison, and vigilance-based decisions. They provide a starting point for understanding adaptive neural network clustering and can be extended to implement the complete algorithms.

## Author

Manogna Kothapalli