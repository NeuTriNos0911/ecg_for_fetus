# Fetal ECG Extraction With Pix2Pix GAN

Deep learning project for extracting fetal ECG signals from maternal ECG recordings using Pix2Pix GANs in the time-frequency domain.

## Project Snapshot

- Biomedical signal processing project focused on non-invasive fetal monitoring.
- Uses a Pix2Pix-style conditional GAN for signal-to-signal translation.
- Converts maternal ECG inputs into time-frequency representations for model training.
- Includes training, testing, data utility, visualization, and model modules.
- Built with PyTorch, NumPy, OpenCV, Visdom, and Weights & Biases support.

## Why This Matters

Fetal ECG extraction is difficult because maternal abdominal ECG recordings contain mixed maternal, fetal, and noise components. This project explores a GAN-based approach that learns to separate fetal ECG patterns from maternal ECG signals in the spectrogram domain.

## Project Structure

```text
.
|-- train.py                 # Training entry point
|-- test.py                  # Inference/testing entry point
|-- requirements.txt         # Python dependencies
|-- checkpoints/             # Included model checkpoint artifacts
|-- models/                  # Pix2Pix model and network definitions
|-- options/                 # Train/test option parsers
|-- util/                    # Data, visualization, and helper utilities
`-- notebooks/               # Exploratory notebook
```

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Train the model:

```bash
python train.py
```

Run inference/testing:

```bash
python test.py
```

## Recruiter Notes

This repository demonstrates applied deep learning, medical signal processing, PyTorch model organization, experiment configuration, and practical GAN usage beyond image generation.
