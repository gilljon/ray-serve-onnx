# ray-serve-onnx

**ray-serve-onnx** provides a collection of starter code and examples to deploy ONNX models using [Ray Serve](https://docs.ray.io/en/latest/serve/index.html). Whether you are exploring ONNX for the first time or looking for best practices in production, this repository will help you get started with a scalable and flexible serving setup on Ray.

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Getting Started](#getting-started)
4. [Usage](#usage)
5. [Examples](#examples)
6. [Performance Tips](#performance-tips)
7. [Contributing](#contributing)
8. [License](#license)

---

## Overview
[ONNX (Open Neural Network Exchange)](https://onnx.ai/) is a popular format for model interoperability, enabling you to easily switch between frameworks without rewriting model code. [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) is a scalable model serving framework that can handle high-throughput inference.

This repository provides:
- Starter code for serving ONNX models with Ray Serve.
- Best practices for model loading, inference, and deployment.
- Example code showing how to customize request handling, scale out to multiple replicas, and monitor performance.

---

## Key Features
- **ONNX Model Deployment**: Easily load and run inference on ONNX models using Ray Serve.
- **Scalability**: Leverage Ray’s distributed architecture to scale out your inference workloads.
- **Customization**: Inject custom logic into request handling with Ray Serve’s flexible deployment APIs.
- **Performance**: Combine ONNX’s efficient runtime with Ray Serve’s built-in scaling and concurrency management.

---

## Getting Started

### Prerequisites
- [Python 3.10+](https://www.python.org/downloads/)
- [Ray 2.40+](https://docs.ray.io/en/latest/) (or a recent version of your choice)
- [onnxruntime](https://github.com/microsoft/onnxruntime) or equivalent ONNX runtime library

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/gilljon/ray-serve-onnx.git
   cd ray-serve-onnx
   ```

2. (Optional) Create and activate a new virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Unix-based systems
   # or
   env\Scripts\activate     # On Windows
   ```

3. Install the required Python packages (via pip):
   ```bash
   pip install -r requirements.txt
   ```
   or via Poetry:
   ```bash
   poetry install --no-root
   ```

---

## Usage
1. **Start Ray**: 
   ```bash
   ray start --head
   ```
   Or just let Ray automatically start in local mode from your Python script (see examples).

2. **Run a serving script**:
   ```bash
   serve run examples.embedding_inference:build
   ```
   This will:
   - Initialize Ray (if not already initialized).
   - Download the ONNX model weights from [onnx-models/all-MiniLM-L12-v2-onnx](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) and cache them in `./onnx-models`
   - Create a Ray Serve deployment.
   - Start the Ray Serve HTTP API on port `8000` to receieve inference requests.

3. **Send an inference request**:
   ```bash
   curl -X POST -H "Content-Type: application/json" \
        -d '{"inputs": ["John Doe works at the store."]}' \
        http://localhost:8000/
   ```
   Adjust the request body based on your model’s input structure.

---

## Examples

### 1. Basic Inference (embedding)
- **File**: `examples/embedding_inference.py`  
- **Description**: Simple demonstration of loading a single ONNX embedding model and serving it through Ray Serve.
- **Command**: `serve run examples.embedding:build`

More examples coming soon. Please feel free to open an issue for specific example requests! :) 

---

## Performance Tips
- **Use ONNX Runtime Execution Providers**: Try GPU providers (CUDA, ROCm) or specialized hardware providers (e.g., TensorRT) to speed up your inference.
- **Batching**: Ray Serve allows request batching to improve throughput. Configure `max_batch_size` and `batch_wait_timeout_s` for your deployment.
- **Parallelization**: Ray allows easy scaling with multiple replicas; each replica can host the ONNX model in memory.
- **Profiling**: Use Ray’s built-in metrics or external profilers to identify bottlenecks (I/O, CPU/GPU usage, etc.).

---

## Contributing
Contributions are welcome! If you want to add new examples, fix bugs, or propose new features:
1. Fork the repository.
2. Create a new branch with your contribution: `git checkout -b feature/my-new-feature`.
3. Commit your changes: `git commit -m "Add some feature"`.
4. Push to the branch: `git push origin feature/my-new-feature`.
5. Create a pull request in this repository describing your changes.

---

## License
This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code in this repository. See the [LICENSE](LICENSE) file for full details.

---

If you encounter any issues or have any questions, please [open an issue](https://github.com/gilljon/ray-serve-onnx/issues). I welcome all feedback and contributions. Happy serving!