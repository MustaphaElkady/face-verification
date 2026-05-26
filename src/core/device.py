import logging
from core.config import settings

logger = logging.getLogger(__name__)


def torch_cuda_available() -> bool:
    """
    Returns True only when USE_GPU=true and PyTorch can see CUDA.
    """
    if not settings.use_gpu:
        return False

    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception as exc:
        logger.warning("PyTorch CUDA check failed, falling back to CPU: %s", exc)
        return False


def torch_device() -> str:
    """
    Device string for PyTorch models.
    """
    return "cuda:0" if torch_cuda_available() else "cpu"


def easyocr_gpu_enabled() -> bool:
    """
    EasyOCR uses PyTorch, so only enable GPU when torch CUDA works.
    """
    return torch_cuda_available()


def onnxruntime_execution():
    """
    Returns:
        providers, ctx_id, device_label

    For InsightFace / ONNXRuntime:
        CPU -> providers=["CPUExecutionProvider"], ctx_id=-1
        GPU -> providers=["CUDAExecutionProvider", "CPUExecutionProvider"], ctx_id=0
    """
    if not settings.use_gpu:
        return ["CPUExecutionProvider"], -1, "CPU"

    try:
        import onnxruntime as ort

        # Helps ONNXRuntime find CUDA/cuDNN libraries when available.
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception as exc:
                logger.warning("onnxruntime.preload_dlls() failed: %s", exc)

        available = ort.get_available_providers()

        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0, "GPU"

        logger.warning(
            "USE_GPU=true but CUDAExecutionProvider is not available. "
            "Available ONNX providers: %s. Falling back to CPU.",
            available,
        )
        return ["CPUExecutionProvider"], -1, "CPU"

    except Exception as exc:
        logger.warning("ONNXRuntime GPU check failed, falling back to CPU: %s", exc)
        return ["CPUExecutionProvider"], -1, "CPU"