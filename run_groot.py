import argparse
import os
from typing import Dict
import numpy as np
from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values
from gr00t.data.dataset import LeRobotSingleDataset
import torch 
import torch_tensorrt
from contextlib import nullcontext
from transformers.modeling_outputs import BaseModelOutputWithPooling
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from functools import partial
from typing import Any, Optional
from utils import benchmark_policy, compare_benchmark_outputs

def get_groot_policy(args: argparse.Namespace):
    """
    Get the Groot policy. Change the attention implementation to SDPA.
    compute_dtype is set to the precision specified in the args.
    Args:
        args: The arguments for the policy
    Returns:
        The Groot policy
    """
    with torch.inference_mode(), torch.no_grad():
        # Load the policy
        data_config = DATA_CONFIG_MAP["fourier_gr1_arms_only"]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=args.embodiment_tag,
            modality_config=modality_config,
            modality_transform=modality_transform,
            device=args.device,
            denoising_steps=args.denoising_steps,
            compute_dtype=get_torch_dtype(args.precision),
        )
        # Cast all the model components of the policy to the precision specified in the args.
        policy.model = policy.model.eval().to(get_torch_dtype(args.precision))
        # We don't support flash_attn in the current version of Torch-TensorRT. So we replace it with SDPA in the model.
        policy.model.backbone.eagle_model.vision_model.config._attn_implementation = "sdpa"
        policy.model.backbone.eagle_model.language_model.config._attn_implementation = "sdpa"
    
    return policy

def get_dataset(args: argparse.Namespace):
    """
    Get the dataset.
    """
    with torch.inference_mode(), torch.no_grad():
        data_config = DATA_CONFIG_MAP["fourier_gr1_arms_only"]
        modality_config = data_config.modality_config()
        # load the dataset
        dataset = LeRobotSingleDataset(
            dataset_path=args.dataset_path,
            modality_configs=modality_config,
            video_backend="decord",
            video_backend_kwargs=None,
            transforms=None,  # We'll handle transforms separately through the policy
            embodiment_tag=args.embodiment_tag,
        )

        step_data = dataset[0]
        return step_data

def eval_outputs(pyt_model, trt_model, inputs, args: argparse.Namespace): 
    """
    Evaluate the outputs and print the difference between the PyTorch and Torch-TensorRT models.

    Args:
        pyt_model: The PyTorch model
        trt_model: The Torch-TensorRT model
        inputs: The inputs to the models
    Returns:
        None
    """
    if args.eval:
        if isinstance(pyt_model, torch.nn.Module):
            pyt_model = pyt_model.to(args.device)

        if isinstance(inputs, (tuple, list)):
            pyt_output = pyt_model(*inputs)
            trt_output = trt_model(*inputs)
        elif isinstance(inputs, dict):
            pyt_output = pyt_model(**inputs)
            trt_output = trt_model(**inputs)
        else:
            pyt_output = pyt_model(inputs)
            trt_output = trt_model(inputs)

        if isinstance(pyt_output, torch.Tensor) and isinstance(trt_output, torch.Tensor):
            print("Diff: ", torch.mean(torch.abs(pyt_output - trt_output)))
        elif isinstance(pyt_output, BaseModelOutputWithPooling) and isinstance(trt_output, BaseModelOutputWithPooling):
            print("Diff: ", torch.mean(torch.abs(pyt_output[1][-1] - trt_output[1][-1])))
            print("Diff: ", torch.mean(torch.abs(pyt_output[0] - trt_output[0])))

def compile_vision_model(vision_model: torch.nn.Module, args: argparse.Namespace):
    """
    Compile the vision model of the eagle backbone using Torch-TensorRT.
    """
    BATCH_SIZE = 2
    NUM_CHANNELS = vision_model.config.num_channels
    IMAGE_SIZE = vision_model.config.image_size
    pixel_values = torch.randn(
        (BATCH_SIZE, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
        dtype=get_torch_dtype(args.precision),
        device=args.device,
    )
    kwarg_inputs = {
        "pixel_values": pixel_values,
        "output_hidden_states": False,
        # "return_dict": True,
    }
    BATCH_DIM = torch.export.Dim("batch", min=1, max=8)
    kwarg_dynamic_shapes = {
        "pixel_values": {0: BATCH_DIM},
        "output_hidden_states": None,
        # "return_dict": None,
    }

    settings = get_compilation_args(args)
    settings.update({"allow_complex_guards_as_runtime_asserts": True})
    trt_vision_model = torch_tensorrt.MutableTorchTensorRTModule(vision_model, **settings)
    trt_vision_model.set_expected_dynamic_shape_range((), kwarg_dynamic_shapes)
    with (torch_tensorrt.dynamo.Debugger() if args.debug else nullcontext()):
        trt_vision_model(**kwarg_inputs)
    
    eval_outputs(vision_model, trt_vision_model, kwarg_inputs, args)

    if args.benchmark and args.fn_name == "vision_model":
        pyt_timings = benchmark_policy(vision_model, (), kwarg_inputs, args=args)
        trt_timings = benchmark_policy(trt_vision_model, (), kwarg_inputs, args=args)
        compare_benchmark_outputs(pyt_timings, trt_timings)

    return trt_vision_model

def compile_language_model(language_model: torch.nn.Module, args: argparse.Namespace, attention_mask: Optional[torch.Tensor] = None):
    """
    Compile the language model of the eagle backbone using Torch-TensorRT.
    """
    BATCH_SIZE = 2
    SEQ_LEN = attention_mask.shape[1] if attention_mask is not None else 294
    HIDDEN_SIZE = 2048
    inputs_embeds = torch.randn((BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE), dtype=get_torch_dtype(args.precision), device=args.device)
    position_ids = torch.arange(SEQ_LEN, dtype=torch.int64, device=args.device).unsqueeze(0).repeat(BATCH_SIZE, 1)
    kwarg_inputs = {
        "inputs_embeds": inputs_embeds,
        "position_ids": position_ids,
        "output_hidden_states": True,
    }

    BATCH_DIM = torch.export.Dim("batch", min=1, max=8)
    SEQ_LEN_DIM = torch.export.Dim("seq_len", min=1, max=1024)
    kwarg_dynamic_shapes = {
        "inputs_embeds": {0: BATCH_DIM, 1: SEQ_LEN_DIM},
        "position_ids": {0: BATCH_DIM, 1: SEQ_LEN_DIM},
        "output_hidden_states": None,
    }

    settings = get_compilation_args(args)
    
    trt_language_model = torch_tensorrt.MutableTorchTensorRTModule(language_model, **settings)
    trt_language_model.set_expected_dynamic_shape_range((), kwarg_dynamic_shapes)
    with (torch_tensorrt.dynamo.Debugger() if args.debug else nullcontext()):
        trt_language_model(**kwarg_inputs)
    
    eval_outputs(language_model, trt_language_model, kwarg_inputs, args)

    if args.benchmark and args.fn_name == "language_model":
        pyt_timings = benchmark_policy(language_model, (), kwarg_inputs, args=args)
        trt_timings = benchmark_policy(trt_language_model, (), kwarg_inputs, args=args)
        compare_benchmark_outputs(pyt_timings, trt_timings)

    return trt_language_model

def compile_eagle_backbone(eagle_backbone: torch.nn.Module, args: argparse.Namespace, attention_mask: Optional[torch.Tensor] = None):
    """
    Compile the eagle backbone's vision model and language model separately using Torch-TensorRT.

    Args:
        eagle_backbone: The eagle backbone of the GR00T model
        args: The arguments for the compilation

    Returns:
        The compiled eagle backbone
    """
    eagle_backbone.vision_model.config._attn_implementation = "sdpa"
    eagle_backbone.language_model.config._attn_implementation = "sdpa"

    # Compile the vision model
    trt_vision_model = compile_vision_model(eagle_backbone.vision_model, args)
    eagle_backbone.vision_model = trt_vision_model

    # Compile the language model
    trt_language_model = compile_language_model(eagle_backbone.language_model, args, attention_mask=attention_mask)
    eagle_backbone.language_model = trt_language_model

    return eagle_backbone

class VLComponents(torch.nn.Module):
    """
    A wrapper class that combines vision-language layer normalization and self-attention components.
    """
    def __init__(self, vlln, vl_self_attention):
        super().__init__()
        self.vlln = vlln
        self.vl_self_attention = vl_self_attention
    
    def forward(self, x):
        x = self.vlln(x)
        x = self.vl_self_attention(x)
        return x

def optimized_process_backbone_output(self, backbone_output: Any) -> Any:
    """
    Optimized process backbone output for TensorRT optimization.
    """
    backbone_features = backbone_output["backbone_features"]
    backbone_features = self.trt_vl_components_module(backbone_features)
    backbone_output["backbone_features"] = backbone_features
    return backbone_output

def compile_vl_components(model: torch.nn.Module, args: argparse.Namespace, attention_mask: Optional[torch.Tensor] = None):
    """
    Compile the vision-language layer normalization and self-attention components
    of the action head for TensorRT optimization.
    
    Args:
        model: The GR00T model containing the action head
        device: Device to run compilation on (default: "cuda")
        
    Returns:
        Tuple of compiled TensorRT versions of the vlln and vl_self_attention components
    """
    batch_size = 2
    hidden_dim = model.config.backbone_embedding_dim
    seq_len = attention_mask.shape[1] if attention_mask is not None else 294
    inputs = torch.randn(
        (batch_size, seq_len, hidden_dim),
        dtype=get_torch_dtype(args.precision),
        device=args.device,
    )
    BATCH_DIM = torch.export.Dim("batch", min=1, max=8)
    SEQ_LEN_DIM = torch.export.Dim("seq_len", min=1, max=1024)
    dynamic_shapes = ({0: BATCH_DIM, 1: SEQ_LEN_DIM},)

    vl_components_module = VLComponents(model.vlln, model.vl_self_attention)
    # Compile the vlln
    trt_vl_components_module = torch_tensorrt.MutableTorchTensorRTModule(vl_components_module, **get_compilation_args(args))

    trt_vl_components_module.set_expected_dynamic_shape_range(dynamic_shapes, {})
    with (torch_tensorrt.dynamo.Debugger() if args.debug else nullcontext()):
        trt_vl_components_module(inputs)

    # TODO: Fix the eval_outputs function to handle the VLComponents module
    # eval_outputs(model.process_backbone_output, trt_vl_components_module, inputs, args)

    # Add the compiled VLComponents module to the model
    model.trt_vl_components_module = trt_vl_components_module

    # Patch the process_backbone_output function to use the compiled VLComponents module
    model.process_backbone_output = partial(
            optimized_process_backbone_output, model
        )

    if args.benchmark and args.fn_name == "vl_components":
        pyt_timings = benchmark_policy(vl_components_module, (inputs,), {}, args=args)
        trt_timings = benchmark_policy(trt_vl_components_module, (inputs,), {}, args=args)
        compare_benchmark_outputs(pyt_timings, trt_timings)

    return model

def compile_state_encoder(model: torch.nn.Module, args: argparse.Namespace, state: Optional[torch.Tensor] = None):
    """
    Compile the state encoder of the GrootN1.5 model using Torch-TensorRT.
    
    Args:
        model: The action head of the GR00T model
        device: Device to run compilation on (default: "cuda")
    """
    BATCH_SIZE = 2
    H, W = 1, 64
    if state is not None:
        H, W = state.shape[1], state.shape[2]
    action_input_state = torch.randn(
        (BATCH_SIZE, H, W),
        dtype=get_torch_dtype(args.precision),
        device=args.device,
    )

    embodiment_id = torch.tensor([24, 24], dtype=torch.int64, device=args.device)
    BATCH_DIM = torch.export.Dim("batch", min=1, max=8)
    dynamic_shapes = ({0: BATCH_DIM}, {0: BATCH_DIM})
    trt_state_encoder = torch_tensorrt.MutableTorchTensorRTModule(model.state_encoder, **get_compilation_args(args))
    trt_state_encoder.set_expected_dynamic_shape_range(dynamic_shapes, {})
    with (torch_tensorrt.dynamo.Debugger() if args.debug else nullcontext()):
        trt_state_encoder(action_input_state, embodiment_id)

    eval_outputs(model.state_encoder, trt_state_encoder, (action_input_state, embodiment_id), args)

    if args.benchmark and args.fn_name == "state_encoder":
        pyt_timings = benchmark_policy(model.state_encoder, (action_input_state, embodiment_id), {}, args=args)
        trt_timings = benchmark_policy(trt_state_encoder, (action_input_state, embodiment_id), {}, args=args)
        compare_benchmark_outputs(pyt_timings, trt_timings)

    return trt_state_encoder

def compile_action_encoder(model: torch.nn.Module, args: argparse.Namespace):
    """
    Compile the action encoder of the GrootN1.5 model using Torch-TensorRT.
    
    Args:
        model: The action head of the GR00T model
        device: Device to run compilation on (default: "cuda")
    """
    BATCH_SIZE = 2
    action_inputs = torch.randn(
        (BATCH_SIZE, model.config.action_horizon, model.config.action_dim),
        dtype=get_torch_dtype(args.precision),
        device=args.device,
    )
    timesteps = torch.tensor([0, 0], dtype=torch.int64, device=args.device)
    embodiment_id = torch.tensor([24, 24], dtype=torch.int64, device=args.device)
    BATCH_DIM = torch.export.Dim("batch", min=1, max=8)
    dynamic_shapes = ({0: BATCH_DIM}, {0: BATCH_DIM}, {0: BATCH_DIM})
    trt_action_encoder = torch_tensorrt.MutableTorchTensorRTModule(model.action_encoder, **get_compilation_args(args))
    trt_action_encoder.set_expected_dynamic_shape_range(dynamic_shapes, {})
    with (torch_tensorrt.dynamo.Debugger() if args.debug else nullcontext()):
        trt_action_encoder(action_inputs, timesteps, embodiment_id)
    eval_outputs(model.action_encoder, trt_action_encoder, (action_inputs, timesteps, embodiment_id), args)

    if args.benchmark and args.fn_name == "action_encoder":
        pyt_timings = benchmark_policy(model.action_encoder, (action_inputs, timesteps, embodiment_id), {}, args=args)
        trt_timings = benchmark_policy(trt_action_encoder, (action_inputs, timesteps, embodiment_id), {}, args=args)
        compare_benchmark_outputs(pyt_timings, trt_timings)

    return trt_action_encoder

def compile_dit_model(model: torch.nn.Module, args: argparse.Namespace, attention_mask: Optional[torch.Tensor] = None, state: Optional[torch.Tensor] = None):
    """
    Compile the DIT model for TensorRT optimization.
    
    Args:
        model: The DIT model to compile
        device: Device to run compilation on (default: "cuda")
    """
    BATCH_SIZE = 2
    # Shape is (batch_size, 49, 1536)
    hidden_states = torch.randn(
        (
            BATCH_SIZE,
            state.shape[1]
            + model.config.action_horizon
            + model.config.num_target_vision_tokens,
            model.config.input_embedding_dim,
        ),
        dtype=get_torch_dtype(args.precision),
        device=args.device,
    )
    seq_len = torch.export.Dim("seq_len", min=1, max=1024)
    batch_dim = torch.export.Dim("batch", min=1, max=8)
    # Shape is (batch_size, 294, 2048)
    encoder_hidden_states = torch.randn(
        (BATCH_SIZE, attention_mask.shape[1], model.config.backbone_embedding_dim),
        dtype=get_torch_dtype(args.precision),
        device=args.device,
    )

    timestep = torch.tensor([0, 0], dtype=torch.int64, device=args.device)
    kwarg_inputs = {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
    }
    kwarg_dynamic_shapes = {
        "hidden_states": {0: batch_dim},
        "encoder_hidden_states": {0: batch_dim, 1: seq_len},
        "timestep": {0: batch_dim},
    }
    trt_dit_model = torch_tensorrt.MutableTorchTensorRTModule(model.model, **get_compilation_args(args))
    trt_dit_model.set_expected_dynamic_shape_range((), kwarg_dynamic_shapes)
    with (torch_tensorrt.dynamo.Debugger() if args.debug else nullcontext()):
        trt_dit_model(**kwarg_inputs)

    eval_outputs(model.model, trt_dit_model, kwarg_inputs, args)

    if args.benchmark and args.fn_name == "dit_model":
        pyt_timings = benchmark_policy(model.model, (), kwarg_inputs, args=args)
        trt_timings = benchmark_policy(trt_dit_model, (), kwarg_inputs, args=args)
        compare_benchmark_outputs(pyt_timings, trt_timings)

    return trt_dit_model

def compile_action_decoder(model: torch.nn.Module, args: argparse.Namespace, state: Optional[torch.Tensor] = None):
    """
    Compile the action decoder of the GrootN1.5 model using Torch-TensorRT.
    """
    BATCH_SIZE = 2
    # Shape is (batch_size, 49, 1024)
    hidden_states = torch.randn(
        (
            BATCH_SIZE,
            state.shape[1]
            + model.config.action_horizon
            + model.config.num_target_vision_tokens,
            model.config.hidden_size,
        ),
        dtype=get_torch_dtype(args.precision),
        device=args.device,
    )
    batch_dim = torch.export.Dim("batch", min=1, max=8)
    embodiment_id = torch.tensor([24, 24], dtype=torch.int64, device=args.device)
    dynamic_shapes = ({0: batch_dim},{0: batch_dim})

    trt_action_decoder = torch_tensorrt.MutableTorchTensorRTModule(model.action_decoder, **get_compilation_args(args))
    trt_action_decoder.set_expected_dynamic_shape_range(dynamic_shapes, {})
    with (torch_tensorrt.dynamo.Debugger() if args.debug else nullcontext()):
        trt_action_decoder(hidden_states, embodiment_id)
    
    eval_outputs(model.action_decoder, trt_action_decoder, (hidden_states, embodiment_id), args)

    if args.benchmark and args.fn_name == "action_decoder":
        pyt_timings = benchmark_policy(model.action_decoder, (hidden_states, embodiment_id), {}, args=args)
        trt_timings = benchmark_policy(trt_action_decoder, (hidden_states, embodiment_id), {}, args=args)
        compare_benchmark_outputs(pyt_timings, trt_timings)

    return trt_action_decoder

def compile_action_head(action_head: torch.nn.Module, args: argparse.Namespace, attention_mask: Optional[torch.Tensor] = None, state: Optional[torch.Tensor] = None):

    # Compile the VL Layernorm and Self-Attention
    action_head = compile_vl_components(action_head, args, attention_mask=attention_mask)

    # Compile the state encoder
    trt_state_encoder = compile_state_encoder(action_head, args, state=state)
    action_head.state_encoder = trt_state_encoder

    # Compile the action encoder
    trt_action_encoder = compile_action_encoder(action_head, args)
    action_head.action_encoder = trt_action_encoder

    # Compile the DIT model
    trt_dit_model = compile_dit_model(action_head, args, attention_mask=attention_mask, state=state)
    action_head.model = trt_dit_model

    # # Compile the action decoder
    trt_action_decoder = compile_action_decoder(action_head, args, state=state)      
    action_head.action_decoder = trt_action_decoder

    return action_head


def get_torch_dtype(precision: str):
    """
    Convert a precision string ("FP16", "FP32", "BF16") to the corresponding torch dtype.
    """
    precision = precision.upper()
    if precision == "FP16":
        return torch.float16
    elif precision == "FP32":
        return torch.float32
    elif precision == "BF16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported precision: {precision}")

def get_compilation_args(args: argparse.Namespace):
    """
    Get the enabled precisions for the Torch-TensorRT compilation.
    """
    enabled_precisions={torch.float32}
    if args.use_explicit_typing:
        enabled_precisions={torch.float32}
    else:
        enabled_precisions={get_torch_dtype(args.precision)}
    
    full_compilation_args = {}
    full_compilation_args.update({"enabled_precisions": enabled_precisions})
    full_compilation_args.update({"min_block_size": 1})
    
    if args.disable_tf32:
        full_compilation_args.update({"disable_tf32": args.disable_tf32})

    if args.use_cpp_runtime:
        full_compilation_args.update({"use_python_runtime": not args.use_cpp_runtime})
    else:
        full_compilation_args.update({"use_python_runtime": True})

    if args.use_explicit_typing:
        full_compilation_args.update({"use_explicit_typing": args.use_explicit_typing})
    if args.use_fp32_acc:
        full_compilation_args.update({"use_fp32_acc": args.use_fp32_acc})
    if args.debug:
        full_compilation_args.update({"debug": args.debug})
    if args.cpu_offload:
        full_compilation_args.update({"offload_module_to_cpu": args.cpu_offload})
    
    full_compilation_args.update({"allow_complex_guards_as_runtime_asserts": True})
    full_compilation_args.update({"strict": False})
    full_compilation_args.update({"require_full_compilation": True})
    return full_compilation_args

def compare_predictions(pred_tensorrt, pred_torch):
    """
    Compare the similarity between TensorRT and PyTorch predictions

    Args:
        pred_tensorrt: TensorRT prediction results (numpy array)
        pred_torch: PyTorch prediction results (numpy array)
    """
    print("\n=== Prediction Comparison ===")

    # Ensure both predictions contain the same keys
    assert pred_tensorrt.keys() == pred_torch.keys(), "Prediction keys do not match"

    # Calculate max label width for alignment
    max_label_width = max(
        len("Cosine Similarity (PyTorch/TensorRT):"),
        len("L1 Mean/Max Distance (PyTorch/TensorRT):"),
        len("Max Output Values (PyTorch/TensorRT):"),
        len("Mean Output Values (PyTorch/TensorRT):"),
        len("Min Output Values (PyTorch/TensorRT):"),
    )

    for key in pred_tensorrt.keys():
        tensorrt_array = pred_tensorrt[key]
        torch_array = pred_torch[key]

        # Convert to PyTorch tensors
        tensorrt_tensor = torch.from_numpy(tensorrt_array).to(torch.float32) if isinstance(tensorrt_array, np.ndarray) else tensorrt_array.to(torch.float32)
        torch_tensor = torch.from_numpy(torch_array).to(torch.float32) if isinstance(torch_array, np.ndarray) else torch_array.to(torch.float32)

        # Ensure tensor shapes are the same
        assert (
            tensorrt_tensor.shape == torch_tensor.shape
        ), f"{key} shapes do not match: {tensorrt_tensor.shape} vs {torch_tensor.shape}"

        # Calculate cosine similarity
        flat_tensorrt = tensorrt_tensor.flatten()
        flat_torch = torch_tensor.flatten()

        # Manually calculate cosine similarity
        dot_product = torch.dot(flat_tensorrt, flat_torch)
        norm_tensorrt = torch.norm(flat_tensorrt)
        norm_torch = torch.norm(flat_torch)
        cos_sim = dot_product / (norm_tensorrt * norm_torch)

        # Calculate L1 distance
        l1_dist = torch.abs(flat_tensorrt - flat_torch)

        print(f"\n{key}:")
        print(f'{"Cosine Similarity (PyTorch/TensorRT):".ljust(max_label_width)} {cos_sim.item()}')
        print(
            f'{"L1 Mean/Max Distance (PyTorch/TensorRT):".ljust(max_label_width)} {l1_dist.mean().item():.4f}/{l1_dist.max().item():.4f}'
        )
        print(
            f'{"Max Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.max().item():.4f}/{tensorrt_tensor.max().item():.4f}'
        )
        print(
            f'{"Mean Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.mean().item():.4f}/{tensorrt_tensor.mean().item():.4f}'
        )
        print(
            f'{"Min Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.min().item():.4f}/{tensorrt_tensor.min().item():.4f}'
        )

def get_input_info(policy, observations):
    is_batch = policy._check_state_is_batched(observations)
    if not is_batch:
        observations = unsqueeze_dict_values(observations)

    normalized_input = unsqueeze_dict_values
    # Apply transforms
    normalized_input = policy.apply_transforms(observations)

    return normalized_input["eagle_attention_mask"], normalized_input["state"]

def run_groot_inference(
    args: argparse.Namespace
) -> Dict[str, float]:

    with torch.inference_mode(), torch.no_grad():
        # Get the Groot policy
        policy = get_groot_policy(args)
        # Provides fixed actions inputs to action encoder for deterministic inference
        # ensure PyTorch and TensorRT have the same init_actions
        if not hasattr(policy.model.action_head, "init_actions"):
            policy.model.action_head.init_actions = torch.randn(
                (1, policy.model.action_head.action_horizon, policy.model.action_head.action_dim),
                dtype=get_torch_dtype(args.precision),
                device=args.device,
            )
        from deployment_scripts.action_head_utils import action_head_pytorch_forward
        policy.model.action_head.get_action = partial(
            action_head_pytorch_forward, policy.model.action_head
        )
        step_data = get_dataset(args)

        # Set the model to evaluation mode and move it to the specified device with the correct precision
        model=policy.model.eval().to(args.device).to(get_torch_dtype(args.precision))

        # Run pytorch inference and get the predicted action
        pyt_predicted_action = policy.get_action(step_data) 
        attention_mask, state = get_input_info(policy, step_data)
        

        if args.fn_name == "eagle_backbone":
            trt_eagle_backbone = compile_eagle_backbone(model.backbone.eagle_model, args, attention_mask=attention_mask)
            model.backbone.eagle_model = trt_eagle_backbone
        elif args.fn_name == "vision_model":
            trt_vision_model = compile_vision_model(model.backbone.eagle_model.vision_model, args)
            model.backbone.eagle_model.vision_model = trt_vision_model
        elif args.fn_name == "language_model":
            trt_language_model = compile_language_model(model.backbone.eagle_model.language_model, args, attention_mask=attention_mask)
            model.backbone.eagle_model.language_model = trt_language_model
        elif args.fn_name == "vl_components":
            trt_vl_components = compile_vl_components(model.action_head, args, attention_mask=attention_mask)
            model.action_head = trt_vl_components
        elif args.fn_name == "state_encoder":
            trt_state_encoder = compile_state_encoder(model.action_head, args, state=state)
            model.action_head.state_encoder = trt_state_encoder
        elif args.fn_name == "action_encoder":
            trt_action_encoder = compile_action_encoder(model.action_head, args)
            model.action_head.action_encoder = trt_action_encoder
        elif args.fn_name == "dit_model":
            trt_dit_model = compile_dit_model(model.action_head, args, attention_mask=attention_mask, state=state)
            model.action_head.model = trt_dit_model
        elif args.fn_name == "action_decoder":
            trt_action_decoder = compile_action_decoder(model.action_head, args, state=state)
            model.action_head.action_decoder = trt_action_decoder
        elif args.fn_name == "action_head":
            trt_action_head = compile_action_head(model.action_head, args, attention_mask=attention_mask, state=state)
            model.action_head = trt_action_head
        elif args.fn_name == "all":
            if args.benchmark:
                pyt_timings = benchmark_policy(policy.get_action, (step_data,), {}, args=args)

            trt_eagle_backbone = compile_eagle_backbone(model.backbone.eagle_model, args)
            model.backbone.eagle_model = trt_eagle_backbone

            trt_action_head = compile_action_head(model.action_head, args, attention_mask=attention_mask, state=state)
            model.action_head = trt_action_head

            # Replace the model in the policy with the Torch-TensorRT compiled model
            policy.model = model
            # Run the Torch-TensorRT compiled model and get the predicted action
            trt_predicted_action = policy.get_action(step_data, use_position_ids=True)

            if args.benchmark:
                trt_timings = benchmark_policy(policy.get_action, (step_data,), {"use_position_ids": True}, args=args)

            # Evaluate the difference between the PyTorch and Torch-TensorRT models
            compare_predictions(pyt_predicted_action, trt_predicted_action)

            # Compare the performance of the PyTorch and Torch-TensorRT models
            if args.benchmark:
                compare_benchmark_outputs(pyt_timings, trt_timings)

        else:
            print("No component is compiled with Torch-TensorRT. Running PyTorch inference.")


        print("=========Groot N1.5 3B inference complete=========")

    

if __name__ == "__main__":
    # Make sure you have logged in to huggingface using `huggingface-cli login` with your nvidia email.
    parser = argparse.ArgumentParser(description="Run Groot Inference")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset",
        default=os.path.join(os.getcwd(), "demo_data/robot_sim.PickNPlace"),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model",
        default="nvidia/GR00T-N1.5-3B",
    )

    parser.add_argument(
        "--precision",
        type=str,
        help="FP16 or FP32",
        default="FP16",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        help="Number of denoising steps",
        default=4,
    )
    parser.add_argument(
        "--fn_name",
        type=str,
        help="Name of the function to run",
        default="all",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="Embodiment tag",
        default="gr1",
    )
    parser.add_argument(
        "--use_fp32_acc", action="store_true", help="Enable fp32 accumulation (default: False)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode (default: False)"
    )
    parser.add_argument(
        "--use_cpp_runtime", action="store_true", help="Enable cpp runtime (default: False)"
    )
    parser.add_argument(
        "--disable_tf32", action="store_true", help="Disable tf32 (default: False)"
    )
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate the outputs of the model (default: False)"
    )
    parser.add_argument(
        "--cpu_offload", action="store_true", help="Enable cpu offload (default: False)"
    )
    parser.add_argument(
        "--use_explicit_typing", action="store_true", help="Enable explicit typing (default: False)"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        help="cuda_event or python_timer",
        default="cuda_event",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        help="Number of iterations to run for benchmarking",
        default=10,
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        help="Number of warmup iterations to run for benchmarking",
        default=5,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run the model on",
        default="cuda:0",
    )

    args = parser.parse_args()

    print(f"Dataset path: {args.dataset_path}")
    print(f"Model path: {args.model_path}")

    # Run the Groot inference
    run_groot_inference(args)
