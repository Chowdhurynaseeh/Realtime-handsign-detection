# Real-Time Hand Sign Detection

Welcome to the **Real-Time Hand Sign Detection** repository! This project leverages deep learning to detect and recognize various hand signs in real-time, using a custom-trained YOLOX model. The system is designed for applications such as gesture-based interaction, sign language interpretation, and more.
![image](https://github.com/user-attachments/assets/5f26aa01-4a58-4680-823b-fba02b3a0b5a)


## Table of Contents
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Auto Labeling Tool](#auto-labeling-tool)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- **Real-Time Detection**: Uses a YOLOX model for accurate, real-time hand sign detection.
- **Customizable and Scalable**: Includes utilities for custom data labeling, model training, and evaluation.
- **Auto Labeling Tool**: A built-in tool for automatic data labeling to streamline the preparation of training datasets.
- **Python-based Implementation**: Easily customizable and extendable Python code.

---

## Demo
A demo of the hand sign detection in action can be run by following the instructions below. The demo uses the `Ninjutsu_demo.py` script for detecting hand signs in a live video feed.

---

## Installation
To set up this project locally, follow these steps:

1. **Clone the Repository**
    ```bash
    git clone https://github.com/Chowdhurynaseeh/Realtime-handsign-detection.git
    cd Realtime-handsign-detection
    ```

2. **Install Dependencies**
    Make sure to have Python 3.7+ installed. Then, install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download Model Weights**
    Download the pre-trained YOLOX model weights from [YOLOX GitHub](https://github.com/Megvii-BaseDetection/YOLOX) and place them in the `model/yolox` directory.

---
![image](https://github.com/user-attachments/assets/16366a5d-17a7-4564-9ec0-03a16e7d7755)
## Usage
To run the real-time hand sign detection demo:

```bash
python Ninjutsu_demo.py



