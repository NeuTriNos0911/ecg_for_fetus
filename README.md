# ecg_for_fetus
ECG extraction from maternal signals using Pix2Pix GAN in the time-frequency domain.
This repository implements a deep learning-based method for extracting fetal electrocardiogram (fECG) signals from maternal abdominal ECG recordings using Pix2Pix GAN. By working in the time-frequency domain, the model learns a mapping from maternal ECG spectrograms to fetal ECG spectrograms, enabling more accurate separation of fetal signals even in noisy environments.

The project includes scripts for data preparation, training, testing, and evaluation. It is based on the PyTorch implementation of Pix2Pix, and includes configuration for Visdom and W&B for visualization and tracking.

Main Features:

Pix2Pix GAN-based signal-to-signal translation

Time-frequency domain analysis for fECG extraction

Easy training and testing setup

Evaluation against conventional signal processing methods

Use Cases:

Biomedical signal processing

Non-invasive fetal monitoring

GAN applications in healthcare
