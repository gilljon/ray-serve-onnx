import os
from typing import Dict

import onnxruntime as ort
from huggingface_hub import snapshot_download
from ray import serve
from transformers import AutoTokenizer


# Execution providers for ONNX runtime
# ONNX provides fallback mechanisms to run on CPU if GPU is not available
# See CUDAExecutionProvider options: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#configuration-options
EXECUTION_PROVIDERS = [
    ("CUDAExecutionProvider", {"device_id": 0}),
    "CPUExecutionProvider",
]


@serve.deployment
class ONNXInference:
    def __init__(self, model_dir: str):
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # Create an inference session for the ONNX model
        model_path = os.path.join(model_dir, "model.onnx")
        # Create session options for ONNX runtime
        session_options = self._init_session_options()
        self.session = ort.InferenceSession(
            model_path,
            providers=EXECUTION_PROVIDERS,
            sess_options=session_options,
        )

    def _init_session_options(self):
        """Initializes session options for ONNX runtime."""
        # See SessionOptions: https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.SessionOptions
        session_options = ort.SessionOptions()
        session_options.enable_mem_pattern = True
        return session_options

    async def __call__(self, request) -> Dict:
        """Handles incoming requests."""
        # Parse JSON from the request
        data = await request.json()
        input_data = data["inputs"]
        # Tokenize the inputs
        inputs = self.tokenizer(
            input_data,
            return_tensors="np",
            padding=True,
            truncation=True,
        )

        # Run inference using ONNX runtime
        # If your model has multiple inputs, handle them accordingly
        result = self.session.run(None, dict(inputs))

        # Return the result in a JSON-compatible format
        return {"output": result[0].tolist()}


def build(args):
    # Deploy the model with the specified model path
    model = "onnx-models/all-MiniLM-L12-v2-onnx"
    model_subdir = model.replace("/", "--")
    destination_dir = "./onnx-models/"
    model_dir = os.path.join(destination_dir, model_subdir)
    snapshot_download(
        model,
        local_dir=model_dir,
    )
    return ONNXInference.bind(model_dir=model_dir)
