# SeqGAN in Tensorflow

As part of the implementation series of [Joseph Lim's group at USC](http://csail.mit.edu/~lim), our motivation is to accelerate (or sometimes delay) research in the AI community by promoting open-source projects. To this end, we implement state-of-the-art research papers, and publicly share them with concise reports. Please visit our [group github site](https://github.com/gitlimlab) for other projects.

This project is implemented by [Shaofan Lai](https://github.com/shaofanl) and reviewed by [Reviewer's name](Reviewer's url).

## Descriptions
This project includes a [[Tensorflow](https://github.com/tensorflow/tensorflow)] implementation of **SeqGAN** proposed in the paper [[SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/abs/1609.05473)] by Lantao Yu et al. at Shanghai Jiao Tong University and University College London.

[Write the descriptions of your project here, including the problem that we are interested in, the high-level idea of the proposed method, the datasets, etc.]
SeqGAN adapts GAN for sequential generation. It regards the generator as a policy in reinforcement learning and the discriminator is trained to provide the reward. To evaluate unfinished sequences, Monto-Carlo search is also applied to sample the complete sequences.
One thing to notice is that SeqGAN doesn't sample a noise code but includes the variance by randomly selecting tokens during the generation.

<p align="center">
    <img src="https://github.com/LantaoYu/SeqGAN/raw/master/figures/seqgan.png">
</p>

## Prerequisites

- Python 2.7
- [Tensorflow 1.0.0](https://github.com/tensorflow/tensorflow/tree/r1.0)

## Usage

[Describe how to use your codes including how to get datasets, how to use your own datasets (if needed), how to train the models, how to evaluate trained models, etc.]

### Datasets

### Training

### Testing

## Results

[Show results here. It can includes training curves, training time, training accuracy, testing accuracy, generated images, etc.]

### Training

### Testing

## Related works

[Mention related works here]

## Author

Shao-Hua Sun / [@shaohua0116](https://github.com/shaohua0116/) @ [Joseph Lim's research lab](https://github.com/gitlimlab) @ USC
