# The Future is Log-Gaussian: ResNets and Their Infinite-Depth-and-Width Limit at Initialization

Link: https://arxiv.org/pdf/2106.04013.pdf

### Things to check out
- NN as Gaussian process: https://en.wikipedia.org/wiki/Neural_network_Gaussian_process
- NTK (Neural Tangent Kernel): https://arxiv.org/pdf/1806.07572.pdf
- Infinite-width limit theorms
- Mean field limits:

### Key constributions
- Hypoactivation of ResNet
- log-Gaussian behavour of ResNet at initialization depending on d/n ratio
- Monte Carlo simulations verifying the log-Gaussian behaviour at initialization
- Balanced ResNet architecture which reduces the variance of the log-Gaussian behaviour

### Theorems
- Theorem 1: Log-Gaussian behaviour of ResNet at initialization
- Proposition 2: Hypoactivation of  ResNet
- Theorem 4: Log-Gaussian behaviour of Balanced ResNet at initialization (Variance reduction)
- B.1, B.2 Mean and variance computations
- Unit sphere / Gaussian distribution theorems

### Experiments
- Monte Carlo simulations of ResNet at initialization
- Monte Carlo simulations of Balanced ResNet at initialization
- Training of ResNet and Balanced ResNet on a standard dataset

### Further work
- Can we show something about how the training process changes the distribution of the weights, similar to NTK?
- Changes in the distribution of the weights during training (experimentally)

### Questions
- How exactly does the scaling work? What is the difference between mean field and 1/sqrt(n) scaling? 

### Plan for report
- Abstract
Previous work has focused on infinite width limits, but ResNets were invented to be 
- Introduction
- Background
- Theorems
- Experiments
- Discussion
- Further work
- References