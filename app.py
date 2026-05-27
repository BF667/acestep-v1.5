"""
ACE-Step v1.5 - HuggingFace Space Entry Point
This file serves as the entry point for HuggingFace Space deployment.
It initializes the service and launches the Gradio interface.
ZeroGPU Support:
- ZeroGPU uses the 'spaces' package to intercept CUDA operations
- Models are loaded to "cuda" during startup but actual GPU allocation is deferred
- Handlers are registered globally so forked processes inherit them without pickling
- @spaces.GPU decorators are on top-level Gradio event handlers, not internal functions
- nano-vllm uses direct CUDA APIs that bypass spaces interception, so we use PyTorch backend
"""
import os
import sys

# Get current directory (app.py location)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add nano-vllm to Python path (local package)
nano_vllm_path = os.path.join(current_dir, "acestep", "third_parts", "nano-vllm")
if os.path.exists(nano_vllm_path):
    sys.path.insert(0, nano_vllm_path)

# Disable Gradio analytics
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Clear proxy settings that may affect Gradio
for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    os.environ.pop(proxy_var, None)

# Import spaces for ZeroGPU support (must be imported before torch for proper interception)
# This is a no-op if not running on HuggingFace Spaces
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

import torch
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.dataset_handler import DatasetHandler
from acestep.gradio_ui import create_gradio_interface
from acestep.quiet_loading import quiet_mode, quiet_print

# Detect ZeroGPU environment
IS_HUGGINGFACE_SPACE = os.environ.get("SPACE_ID") is not None
# ZeroGPU detection: check env var OR assume ZeroGPU for all HF Spaces (safer default)
# The SPACE_HARDWARE env var is unreliable, so we assume ZeroGPU if on HF Space
IS_ZEROGPU = IS_HUGGINGFACE_SPACE or os.environ.get("ZEROGPU") is not None


def get_gpu_memory_gb():
    """
    Get GPU memory in GB. Returns 0 if no GPU is available.
    """
    try:
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_gb = total_memory / (1024**3)
            return memory_gb
        else:
            return 0
    except Exception as e:
        print(f"Warning: Failed to detect GPU memory: {e}", file=sys.stderr)
        return 0


def get_persistent_storage_path():
    """
    Detect and return a writable persistent storage path.
    
    HuggingFace Space persistent storage requirements:
    1. Must be enabled in Space settings
    2. Path is typically /data for Docker SDK
    3. Falls back to app directory if /data is not writable
    
    Local development:
    - Set CHECKPOINT_DIR environment variable to use local checkpoints
      Example: CHECKPOINT_DIR=/path/to/checkpoints python app.py
      The path should be the parent directory of 'checkpoints' folder
    """
    # Check for local checkpoint directory override (for development)
    checkpoint_dir_override = os.environ.get("CHECKPOINT_DIR")
    if checkpoint_dir_override:
        # If user specifies the checkpoints folder directly, use its parent
        if checkpoint_dir_override.endswith("/checkpoints") or checkpoint_dir_override.endswith("\\checkpoints"):
            checkpoint_dir_override = os.path.dirname(checkpoint_dir_override)
        if os.path.exists(checkpoint_dir_override):
            print(f"Using local checkpoint directory (CHECKPOINT_DIR): {checkpoint_dir_override}")
            return checkpoint_dir_override
        else:
            print(f"Warning: CHECKPOINT_DIR path does not exist: {checkpoint_dir_override}")
    
    # Try HuggingFace Space persistent storage first
    hf_data_path = "/data"
    
    # Check if /data exists and is writable
    if os.path.exists(hf_data_path):
        try:
            test_file = os.path.join(hf_data_path, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"Using HuggingFace persistent storage: {hf_data_path}")
            return hf_data_path
        except (PermissionError, OSError) as e:
            print(f"Warning: /data exists but is not writable: {e}")
    
    # Fall back to app directory (non-persistent but works without special config)
    fallback_path = os.path.join(current_dir, "data")
    os.makedirs(fallback_path, exist_ok=True)
    print(f"Using local storage (non-persistent): {fallback_path}")
    print("Note: To enable persistent storage, configure it in HuggingFace Space settings")
    return fallback_path


def main():
    """Main entry point for HuggingFace Space"""
    
    # Check for DEBUG_UI mode (skip model initialization for UI development)
    debug_ui = os.environ.get("DEBUG_UI", "").lower() in ("1", "true", "yes")
    if debug_ui:
        print("=" * 60)
        print("DEBUG_UI mode enabled - skipping model initialization")
        print("UI will be fully functional but generation is disabled")
        print("=" * 60)
    
    # Log ZeroGPU detection
    if IS_ZEROGPU:
        quiet_print("=" * 60)
        quiet_print("ZeroGPU environment detected")
        quiet_print("- Using spaces package for GPU allocation")
        quiet_print("- PyTorch backend forced for LLM (nano-vllm incompatible)")
        quiet_print("- GPU will be allocated on-demand during generation")
        quiet_print("=" * 60)
    
    # Get persistent storage path (auto-detect)
    persistent_storage_path = get_persistent_storage_path()
    
    # Detect GPU memory for auto-configuration
    # Note: In ZeroGPU, GPU may not be available during startup, so this may return 0
    gpu_memory_gb = get_gpu_memory_gb()
    
    # For ZeroGPU, we don't need CPU offload as GPU is allocated dynamically
    if IS_ZEROGPU:
        auto_offload = False
        quiet_print("ZeroGPU: CPU offload disabled (GPU allocated on-demand)")
    else:
        auto_offload = gpu_memory_gb > 0 and gpu_memory_gb < 16
    
    if not debug_ui and not IS_ZEROGPU:
        if auto_offload:
            quiet_print(f"Detected GPU memory: {gpu_memory_gb:.2f} GB (< 16GB)")
            quiet_print("Auto-enabling CPU offload to reduce GPU memory usage")
        elif gpu_memory_gb > 0:
            quiet_print(f"Detected GPU memory: {gpu_memory_gb:.2f} GB (>= 16GB)")
            quiet_print("CPU offload disabled by default")
        else:
            quiet_print("No GPU detected, running on CPU")
    
    # Create handler instances
    quiet_print("Creating handlers...")
    dit_handler = AceStepHandler(persistent_storage_path=persistent_storage_path)
    llm_handler = LLMHandler(persistent_storage_path=persistent_storage_path)
    dataset_handler = DatasetHandler()
    
    # Service mode configuration from environment variables
    config_path = os.environ.get(
        "SERVICE_MODE_DIT_MODEL",
        "acestep-v15-xl-turbo"
    )
    # Second DiT model - default to turbo-shift3 for two-model setup
    config_path_2 = os.environ.get("SERVICE_MODE_DIT_MODEL_2", "acestep-v15-turbo").strip()
    
    lm_model_path = os.environ.get(
        "SERVICE_MODE_LM_MODEL",
        "acestep-5Hz-lm-1.7B"
    )
    # For ZeroGPU, force PyTorch backend (nano-vllm uses direct CUDA APIs)
    if IS_ZEROGPU:
        backend = "pt"
    else:
        backend = os.environ.get("SERVICE_MODE_BACKEND", "vllm")
    device = "auto"
    
    quiet_print(f"Service mode configuration:")
    quiet_print(f"  DiT model 1: {config_path}")
    if config_path_2:
        quiet_print(f"  DiT model 2: {config_path_2}")
    quiet_print(f"  LM model: {lm_model_path}")
    quiet_print(f"  Backend: {backend}")
    quiet_print(f"  Offload to CPU: {auto_offload}")
    quiet_print(f"  DEBUG_UI: {debug_ui}")
    quiet_print(f"  ZeroGPU: {IS_ZEROGPU}")
    
    # Determine flash attention availability
    use_flash_attention = dit_handler.is_flash_attention_available()
    quiet_print(f"  Flash Attention: {use_flash_attention}")
    
    # Initialize models (skip in DEBUG_UI mode)
    init_status = ""
    enable_generate = False
    dit_handler_2 = None
    
    if debug_ui:
        # In DEBUG_UI mode, skip all model initialization
        init_status = "⚠️ DEBUG_UI mode - models not loaded\nUI is functional but generation is disabled"
        enable_generate = False
        quiet_print("Skipping model initialization (DEBUG_UI mode)")
    else:
        # Initialize primary DiT model
        quiet_print(f"Initializing DiT model 1: {config_path}...")
        with quiet_mode(suppress_stdout=True, suppress_logging=True, suppress_tqdm=True):
            init_status, enable_generate = dit_handler.initialize_service(
                project_root=current_dir,
                config_path=config_path,
                device=device,
                use_flash_attention=use_flash_attention,
                compile_model=False,
                offload_to_cpu=auto_offload,
                offload_dit_to_cpu=False
            )
        
        if not enable_generate:
            print(f"Warning: DiT model 1 initialization issue: {init_status}", file=sys.stderr)
        else:
            quiet_print("DiT model 1 initialized successfully")
        
        # Initialize second DiT model if configured
        if config_path_2:
            quiet_print(f"Initializing DiT model 2: {config_path_2}...")
            dit_handler_2 = AceStepHandler(persistent_storage_path=persistent_storage_path)
            
            # Share VAE, text_encoder, and silence_latent from the first handler to save memory
            with quiet_mode(suppress_stdout=True, suppress_logging=True, suppress_tqdm=True):
                init_status_2, enable_generate_2 = dit_handler_2.initialize_service(
                    project_root=current_dir,
                    config_path=config_path_2,
                    device=device,
                    use_flash_attention=use_flash_attention,
                    compile_model=False,
                    offload_to_cpu=auto_offload,
                    offload_dit_to_cpu=False,
                    # Share components from first handler
                    shared_vae=dit_handler.vae,
                    shared_text_encoder=dit_handler.text_encoder,
                    shared_text_tokenizer=dit_handler.text_tokenizer,
                    shared_silence_latent=dit_handler.silence_latent,
                )
            
            if not enable_generate_2:
                print(f"Warning: DiT model 2 initialization issue: {init_status_2}", file=sys.stderr)
                init_status += f"\n⚠️ DiT model 2 failed: {init_status_2}"
            else:
                quiet_print("DiT model 2 initialized successfully")
                init_status += f"\n✅ DiT model 2: {config_path_2}"
        
        # Initialize LM model
        checkpoint_dir = dit_handler._get_checkpoint_dir()
        quiet_print(f"Initializing 5Hz LM: {lm_model_path}...")
        with quiet_mode(suppress_stdout=True, suppress_logging=True, suppress_tqdm=True):
            lm_status, lm_success = llm_handler.initialize(
                checkpoint_dir=checkpoint_dir,
                lm_model_path=lm_model_path,
                backend=backend,
                device=device,
                offload_to_cpu=auto_offload,
                dtype=dit_handler.dtype
            )
        
        if lm_success:
            quiet_print("5Hz LM initialized successfully")
            init_status += f"\n{lm_status}"
        else:
            print(f"Warning: 5Hz LM initialization failed: {lm_status}", file=sys.stderr)
            init_status += f"\n{lm_status}"

    # Build available models list for UI
    available_dit_models = [config_path]
    if config_path_2 and dit_handler_2 is not None:
        available_dit_models.append(config_path_2)
    
    # Prepare initialization parameters for UI
    init_params = {
        'pre_initialized': True,
        'service_mode': True,
        'checkpoint': None,
        'config_path': config_path,
        'config_path_2': config_path_2 if config_path_2 else None,
        'device': device,
        'init_llm': True,
        'lm_model_path': lm_model_path,
        'backend': backend,
        'use_flash_attention': use_flash_attention,
        'offload_to_cpu': auto_offload,
        'offload_dit_to_cpu': False,
        'init_status': init_status,
        'enable_generate': enable_generate,
        'dit_handler': dit_handler,
        'dit_handler_2': dit_handler_2,
        'available_dit_models': available_dit_models,
        'llm_handler': llm_handler,
        'language': 'en',
        'persistent_storage_path': persistent_storage_path,
        'debug_ui': debug_ui,
    }
    
    quiet_print("Service initialization completed!")
    
    # Create Gradio interface with pre-initialized handlers
    quiet_print("Creating Gradio interface...")
    demo = create_gradio_interface(
        dit_handler, 
        llm_handler, 
        dataset_handler, 
        init_params=init_params, 
        language='en'
    )
    
    # Enable queue for multi-user support
    quiet_print("Enabling queue for multi-user support...")
    demo.queue(max_size=20)
    
    # Launch
    quiet_print("Launching server on 0.0.0.0:7860...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()