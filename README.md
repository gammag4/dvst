# LVSM - Large View Synthesis Model

| English | [Português](README_PT.md) |

This is my implementation of the [LVSM](https://haian-jin.github.io/projects/LVSM/) model, together with code to train it using the datasets provided.

This is a Novel View Synthesis model, where given a set of images from a 3D scene with their respective camera properties/poses,
the model aims to generate a new view in the scene, given the camera properties and pose of the target view.

### Training results

Due to the model being heavy and hard to train, we had to reduce it a lot to using the resolution of 32x56.

Some results after training in an RTX 4060 Ti with 8GB vRAM for 1 week:

Rotations in x, y and z axes:

https://github.com/user-attachments/assets/ce782296-1df1-4b93-a4f5-75ea909f551a

https://github.com/user-attachments/assets/a9660bc7-d3b8-452f-820d-f88930663891

https://github.com/user-attachments/assets/e484d603-0002-4a19-b97d-7b058353156e

Translations in x, y and z axes:

https://github.com/user-attachments/assets/fff6cbca-1d8c-433d-a742-2a53dbfcff32

https://github.com/user-attachments/assets/1ac25592-2af5-4f4d-a909-674fd8e9c493

https://github.com/user-attachments/assets/2cd17c85-b676-4dfd-be81-d4385bb02190

A remark:
When training with restricted resources, one should increase the values for the betas a lot.
This creates an averaging effect similar to what GrokFast does.
Without this, the model will not converge.

### DVST

This model was originally an attempt to extend LVSM to dynamic scenes,
called Dynamic View Synthesis Transformer (DVST),
but due to lack of resources, the author stopped working on it for the time being.
It has already some code for training with dynamic scenes and for training using multiple GPUs with DDP, albeit some parts are not complete.
