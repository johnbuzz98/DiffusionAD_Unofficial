# DiffusionAD Unofficial Implementation

This repository contains an unofficial implementation of the paper: "DiffusionAD: Denoising Diffusion for Anomaly Detection" by Zhang, H., Wang, Z., Wu, Z., & Jiang, Y. G. (2023), which can be accessed [here](https://arxiv.org/abs/2303.08730).

## Overview

DiffusionAD introduces a novel technique to improve the anomaly detection task by applying denoising diffusion models. By employing this method, the model can discern between normal and abnormal data effectively, with improved performance on various datasets. This project replicates the research paper's methodology using Python.

## Requirements

To run this project, you'll need the following packages:

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- CUDA for GPU acceleration (optional)

You can install the required packages using the following command:

shellCopy code

`pip install -r requirements.txt`

Note: Please ensure to use the appropriate pip command according to your Python environment setup (it could be `pip`, `pip3`, or `python -m pip`)

## Usage

To run the main script:

shellCopy code

`python main.py`

### Adjusting parameters

You can adjust the model parameters in the `config.py` file to fine-tune the model according to your needs.

## Datasets

The original paper tests the model on multiple datasets. In our project, we have included code to handle the following datasets: (Name the datasets you've implemented here)

Please refer to the `datasets/` directory for more details.

## Structure of the Repository

- `main.py`: The main script to train and evaluate the model.
- `model/`: Contains the PyTorch implementation of the DiffusionAD model.
- `datasets/`: Contains the scripts to load and preprocess the datasets.
- `config.py`: Contains the configuration parameters for the model.
- `requirements.txt`: Contains the required packages to run this project.

## Results

(Name the metrics you're evaluating on here, such as ROC-AUC, Precision@k, etc.) on various datasets are as follows:

|Dataset|Metric1|Metric2|...|
|---|---|---|---|
|Dataset1|x.xx|x.xx|...|
|Dataset2|x.xx|x.xx|...|
|...|...|...|...|

(Note: You might want to add plots/figures that highlight the results if any)

## License

This project is licensed under the MIT License - see the [LICENSE](https://chat.openai.com/LICENSE) file for details.

## Acknowledgements

This repository is an unofficial implementation of the paper: "DiffusionAD: Denoising Diffusion for Anomaly Detection". We would like to express our gratitude to the authors for their invaluable research and contribution to this field.

## References

Zhang, H., Wang, Z., Wu, Z., & Jiang, Y. G. (2023). DiffusionAD: Denoising Diffusion for Anomaly Detection. arXiv preprint arXiv:2303.08730.

## External Code Sources

The Perlin noise code is adopted from [VitjanZ/DRAEM](https://github.com/VitjanZ/DRAEM/blob/main/perlin.py).

## Disclaimer

This project is for educational purposes only. No responsibility is taken for any outcomes that arise from using this code.

**Note**: You can modify the sections as per your project's requirements. Remember to replace placeholders with actual values.