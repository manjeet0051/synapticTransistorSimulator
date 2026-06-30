# IGZO/MgO Synaptic Transistor for Neuromorphic Digit Recognition

A machine learning framework that models an **IGZO/MgO synaptic transistor** and uses its experimentally measured electrical characteristics for **neuromorphic handwritten digit recognition**. The project integrates device-level physics with neural network training by directly utilizing the transistor's potentiation and depression behavior for synaptic weight updates.

---

## Features

* Extraction of device parameters from measured transfer characteristics
* IDS prediction using Polynomial Regression and Neural Networks
* EPSC gain prediction from frequency response data
* Standard MNIST digit recognition for baseline comparison
* Physics-aware learning using experimentally measured IGZO potentiation/depression curves
* Interactive GUI for handwritten digit prediction

---

## Project Structure

```text
igzo_project/
├── data/
│   ├── IDVD_IDBG_IGZO_MgO.csv
│   └── filterchar_IGZO_MgO_new.csv
│
├── models/
│   ├── device_params.pkl
│   ├── poly_model.pkl
│   ├── nn_model.pkl
│   ├── gain_model.pkl
│   ├── igzo_digit_model.pkl
│   └── igzo_TRUE_digit_model.pkl
│
├── 1_train_model.py
├── 2_digit_recognition.py
├── 2b_digit_recognition_IGZO_TRUE.py
├── 3_app.py
├── requirements.txt
└── README.md
```

---

## Dataset Description

### `IDVD_IDBG_IGZO_MgO.csv`

Contains experimentally measured transfer characteristics of the IGZO/MgO synaptic transistor, including:

* Gate voltage (VGS)
* Drain current (IDS)
* Transfer curve information
* Potentiation and depression conductance states

The project utilizes all available conductance states as discrete synaptic weight levels.

### `filterchar_IGZO_MgO_new.csv`

Contains frequency-dependent excitatory post-synaptic current (EPSC) measurements used for gain modeling.

---

## Installation

### Prerequisites

* Python 3.10 or later
* pip package manager

### Clone the Repository

```bash
git clone <repository-url>
cd igzo_project
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Execution Workflow

### Step 1: Train Device Models

```bash
python 1_train_model.py
```

This script:

* Extracts device parameters
* Trains IDS prediction models
* Trains the EPSC gain prediction model
* Saves all generated models in the `models/` directory

Generated files:

```text
models/
├── device_params.pkl
├── poly_model.pkl
├── nn_model.pkl
└── gain_model.pkl
```

---

### Step 2: Standard Digit Recognition (Baseline)

```bash
python 2_digit_recognition.py
```

This model uses conventional gradient-based optimization and serves as a baseline for comparison.

Generated file:

```text
models/igzo_digit_model.pkl
```

---

### Step 2b: TRUE IGZO Neuromorphic Learning (Recommended)

```bash
python 2b_digit_recognition_IGZO_TRUE.py
```

This implementation directly incorporates the experimentally measured transistor behavior into the learning process.

Instead of using the conventional update rule:

```text
W = W − η × ∇L
```

the synaptic weights are updated according to the measured device characteristics:

```text
W = igzo.potentiate(W)
W = igzo.depress(W)
```

This approach enables:

* Physics-aware learning
* Discrete conductance state transitions
* Hardware-realistic synaptic updates
* Improved compatibility with neuromorphic hardware implementation

Generated file:

```text
models/igzo_TRUE_digit_model.pkl
```

---

### Step 3: Launch the GUI Application

```bash
python 3_app.py
```

The application allows users to:

1. Draw a handwritten digit
2. Preprocess the image
3. Run inference using the trained model
4. Display the predicted digit

---

## Models

| Model                       | Description                                 |
| --------------------------- | ------------------------------------------- |
| `poly_model.pkl`            | Polynomial IDS predictor                    |
| `nn_model.pkl`              | Neural-network-based IDS predictor          |
| `gain_model.pkl`            | EPSC gain predictor                         |
| `igzo_digit_model.pkl`      | Standard digit classifier                   |
| `igzo_TRUE_digit_model.pkl` | Physics-aware neuromorphic digit classifier |

---

## Troubleshooting

### ModuleNotFoundError

```bash
pip install -r requirements.txt
```

### Model Files Not Found

Ensure that the following script has been executed first:

```bash
python 1_train_model.py
```

### GUI Does Not Open

```bash
pip install --upgrade pillow
```

### Slow MNIST Download

The MNIST dataset is downloaded only during the first execution and requires an active internet connection.

---

## Research Motivation

Conventional neural network training uses idealized mathematical weight updates that may not accurately represent the behavior of physical synaptic devices. This project bridges the gap between machine learning algorithms and hardware implementation by using experimentally measured characteristics of an IGZO/MgO synaptic transistor to perform synaptic updates.

The resulting framework provides a practical approach toward implementing energy-efficient neuromorphic computing systems based on oxide semiconductor synaptic transistors.

---

## Citation

If you use this project in your research, please cite:

*Low-Temperature Solution-Processed In₂O₃ Synaptic Transistors*, IEEE Transactions on Electron Devices, Vol. 73, No. 1, January 2026.
