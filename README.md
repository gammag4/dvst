# LVSM - Large View Synthesis Model

| English | [Português](README_PT.md) |

This is my implementation of the LVSM model, together with code to train it using the datasets provided.

Due to the model being heavy and hard to train, we had to reduce it a lot to using the resolution of 32x56.

Some results after training in an RTX 4060 Ti with 8GB vRAM for 1 week:

A remark:
When training with restricted resources, increase the values for the betas a lot.
This creates an averaging effect similar to what GrokFast does.
Without this, the model will not converge.


This model was originally an attempt to extend [LVSM](https://haian-jin.github.io/projects/LVSM/) to dynamic scenes,
called Dynamic View Synthesis Transformer (DVST) but due to lack of resources, the author stopped working on it for the time being.
It has already some code for training with dynamic scenes and for training using multiple GPUs with DDP, albeit some parts are not complete.
