# A Mask-based Output Layer for Multi-level Hierarchical Classification
This paper proposes a novel mask-based output layer for multi-level hierarchical classification, addressing the limitations of existing methods which (i) often do not embed the taxonomy structure being used, (ii) use a complex backbone neural network with $n$ disjoint output layers that do not constraint each other,  (iii) may output predictions that are often inconsistent with the taxonomy in place, and (iv) have often a fixed value of $n$.
Specifically, we propose a model agnostic output layer that embeds the taxonomy and that can be combined with any model.
Our proposed output layer implements a top-down divide-and-conquer strategy through a masking mechanism to enforce that predictions comply with the embedded hierarchy structure. Focusing on image classification, we evaluate the performance of our proposed output layer on three different datasets, each with a three-level hierarchical structure. Experiments on these datasets show that our proposed mask-based output layer allows to improve several multi-level hierarchical classification models using various performance metrics.



# A Mask-based Output Layer for Multi-level Hierarchical Classification

- [Click here to open the Notebook in Google Colab ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rbouadjenek/Mask-MLHC/blob/main/Model_Training.ipynb). 

# Published papers related to this project

- Tanya Boone-Sifuentes, Mohamed Reda Bouadjenek, Imran Razzak, Hakim Hacid, and Asef Nazari. [A Mask-based Output Layer for Multi-level Hierarchical Classification](https://rbouadjenek.github.io/papers/sp0060.pdf). Proceedings of the 31st ACM International Conference on Information and Knowledge Management, (CIKM), XXX-XXX, 2022.
- Tanya Boone Sifuentes, Asef Nazari, Imran Razzak, Mohamed Reda Bouadjenek, Antonio Robles-Kelly, Daniel Ierodiaconou, and Elizabeth Oh. [Marine-tree: large-scale marine organisms dataset for hierarchical image classification](https://rbouadjenek.github.io/papers/sp0760.pdf). Proceedings of the 31st ACM International Conference on Information and Knowledge Management, (CIKM), XXX-XXX, 2022.

# Contacts
For more information about this project, please contact:

- Mohamed Reda Bouadjenek: rbouadjenek@gmail.com
