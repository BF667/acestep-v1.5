"""
Gradio UI Generation Section Module
Contains generation section component definitions - Simplified UI
"""
import gradio as gr
from acestep.constants import (
    VALID_LANGUAGES,
    TRACK_NAMES,
    TASK_TYPES_TURBO,
    TASK_TYPES_BASE,
    DEFAULT_DIT_INSTRUCTION,
)
from acestep.gradio_ui.i18n import t


def create_generation_section(dit_handler, llm_handler, init_params=None, language='en') -> dict:
    """Create generation section with simplified UI
    
    Args:
        dit_handler: DiT handler instance
        llm_handler: LM handler instance
        init_params: Dictionary containing initialization parameters and state.
                    If None, service will not be pre-initialized.
        language: UI language code ('en', 'zh', 'ja')
    """
    # Check if service is pre-initialized
    service_pre_initialized = init_params is not None and init_params.get('pre_initialized', False)
    
    # Check if running in service mode (restricted UI)
    service_mode = init_params is not None and init_params.get('service_mode', False)
    
    # Get current language from init_params if available
    current_language = init_params.get('language', language) if init_params else language
    
    # Get available models
    available_dit_models = init_params.get('available_dit_models', []) if init_params else []
    current_model_value = init_params.get('config_path', '') if init_params else ''
    show_model_selector = len(available_dit_models) > 1
    
    with gr.Group():
        # ==================== Service Configuration (Hidden in service mode) ====================
        accordion_open = not service_pre_initialized
        accordion_visible = not service_pre_initialized
        with gr.Accordion(t("service.title"), open=accordion_open, visible=accordion_visible) as service_config_accordion:
            # Language selector at the top
            with gr.Row():
                language_dropdown = gr.Dropdown(
                    choices=[
                        ("English", "en"),
                        ("中文", "zh"),
                        ("日本語", "ja"),
                    ],
                    value=current_language,
                    label=t("service.language_label"),
                    info=t("service.language_info"),
                    scale=1,
                )
            
            with gr.Row(equal_height=True):
                with gr.Column(scale=4):
                    checkpoint_value = init_params.get('checkpoint') if service_pre_initialized else None
                    checkpoint_dropdown = gr.Dropdown(
                        label=t("service.checkpoint_label"),
                        choices=dit_handler.get_available_checkpoints(),
                        value=checkpoint_value,
                        info=t("service.checkpoint_info")
                    )
                with gr.Column(scale=1, min_width=90):
                    refresh_btn = gr.Button(t("service.refresh_btn"), size="sm")
            
            with gr.Row():
                available_models = dit_handler.get_available_acestep_v15_models()
                default_model = "acestep-v15-turbo" if "acestep-v15-turbo" in available_models else (available_models[0] if available_models else None)
                config_path_value = init_params.get('config_path', default_model) if service_pre_initialized else default_model
                config_path = gr.Dropdown(
                    label=t("service.model_path_label"),
                    choices=available_models,
                    value=config_path_value,
                    info=t("service.model_path_info")
                )
                device_value = init_params.get('device', 'auto') if service_pre_initialized else 'auto'
                device = gr.Dropdown(
                    choices=["auto", "cuda", "cpu"],
                    value=device_value,
                    label=t("service.device_label"),
                    info=t("service.device_info")
                )
            
            with gr.Row():
                available_lm_models = llm_handler.get_available_5hz_lm_models()
                default_lm_model = "acestep-5Hz-lm-0.6B" if "acestep-5Hz-lm-0.6B" in available_lm_models else (available_lm_models[0] if available_lm_models else None)
                lm_model_path_value = init_params.get('lm_model_path', default_lm_model) if service_pre_initialized else default_lm_model
                lm_model_path = gr.Dropdown(
                    label=t("service.lm_model_path_label"),
                    choices=available_lm_models,
                    value=lm_model_path_value,
                    info=t("service.lm_model_path_info")
                )
                backend_value = init_params.get('backend', 'vllm') if service_pre_initialized else 'vllm'
                backend_dropdown = gr.Dropdown(
                    choices=["vllm", "pt"],
                    value=backend_value,
                    label=t("service.backend_label"),
                    info=t("service.backend_info")
                )
            
            with gr.Row():
                init_llm_value = init_params.get('init_llm', True) if service_pre_initialized else True
                init_llm_checkbox = gr.Checkbox(
                    label=t("service.init_llm_label"),
                    value=init_llm_value,
                    info=t("service.init_llm_info"),
                )
                flash_attn_available = dit_handler.is_flash_attention_available()
                use_flash_attention_value = init_params.get('use_flash_attention', flash_attn_available) if service_pre_initialized else flash_attn_available
                use_flash_attention_checkbox = gr.Checkbox(
                    label=t("service.flash_attention_label"),
                    value=use_flash_attention_value,
                    interactive=flash_attn_available,
                    info=t("service.flash_attention_info_enabled") if flash_attn_available else t("service.flash_attention_info_disabled")
                )
                offload_to_cpu_value = init_params.get('offload_to_cpu', False) if service_pre_initialized else False
                offload_to_cpu_checkbox = gr.Checkbox(
                    label=t("service.offload_cpu_label"),
                    value=offload_to_cpu_value,
                    info=t("service.offload_cpu_info")
                )
                offload_dit_to_cpu_value = init_params.get('offload_dit_to_cpu', False) if service_pre_initialized else False
                offload_dit_to_cpu_checkbox = gr.Checkbox(
                    label=t("service.offload_dit_cpu_label"),
                    value=offload_dit_to_cpu_value,
                    info=t("service.offload_dit_cpu_info")
                )
            
            init_btn = gr.Button(t("service.init_btn"), variant="primary", size="lg")
            init_status_value = init_params.get('init_status', '') if service_pre_initialized else ''
            init_status = gr.Textbox(label=t("service.status_label"), interactive=False, lines=3, value=init_status_value)
            
            # LoRA Configuration Section
            gr.HTML("<hr><h4>🔧 LoRA Adapter</h4>")
            with gr.Row():
                lora_path = gr.Textbox(
                    label="LoRA Path",
                    placeholder="./lora_output/final/adapter",
                    info="Path to trained LoRA adapter directory",
                    scale=3,
                )
                load_lora_btn = gr.Button("📥 Load LoRA", variant="secondary", scale=1)
                unload_lora_btn = gr.Button("🗑️ Unload", variant="secondary", scale=1)
            with gr.Row():
                use_lora_checkbox = gr.Checkbox(
                    label="Use LoRA",
                    value=False,
                    info="Enable LoRA adapter for inference",
                    scale=1,
                )
                lora_status = gr.Textbox(
                    label="LoRA Status",
                    value="No LoRA loaded",
                    interactive=False,
                    scale=2,
                )
        
        # ==================== Model Selector (Top, only when multiple models) ====================
        with gr.Row(visible=show_model_selector):
            dit_model_selector = gr.Dropdown(
                choices=available_dit_models,
                value=current_model_value,
                label="models",
                scale=1,
            )
        
        # Hidden dropdown when only one model (for event handler compatibility)
        if not show_model_selector:
            dit_model_selector = gr.Dropdown(
                choices=available_dit_models if available_dit_models else [current_model_value],
                value=current_model_value,
                visible=False,
            )
        
        # ==================== Generation Mode (5 modes) ====================
        gr.HTML("<div style='background: #4a5568; color: white; padding: 8px 16px; border-radius: 4px; font-weight: bold;'>Generation Mode</div>")
        with gr.Row():
            generation_mode = gr.Radio(
                choices=[
                    ("Simple", "simple"),
                    ("Custom", "custom"),
                    ("Cover", "cover"),
                    ("Remix", "remix"),
                    ("Repaint", "repaint"),
                ],
                value="custom",
                label="",
                show_label=False,
            )
        
        # ==================== Simple Mode Group ====================
        with gr.Column(visible=False) as simple_mode_group:
            # Row: Song Description + Vocal Language + Random button
            with gr.Row(equal_height=True):
                simple_query_input = gr.Textbox(
                    label=t("generation.simple_query_label"),
                    placeholder=t("generation.simple_query_placeholder"),
                    lines=2,
                    info=t("generation.simple_query_info"),
                    scale=10,
                )
                simple_vocal_language = gr.Dropdown(
                    choices=VALID_LANGUAGES,
                    value="unknown",
                    allow_custom_value=True,
                    label=t("generation.simple_vocal_language_label"),
                    interactive=True,
                    info="use unknown for instrumental",
                    scale=2,
                )
                with gr.Column(scale=1, min_width=60):
                    random_desc_btn = gr.Button(
                        "🎲",
                        variant="primary",
                        size="lg",
                    )
            
            # Hidden components (kept for compatibility but not shown)
            simple_instrumental_checkbox = gr.Checkbox(
                label=t("generation.instrumental_label"),
                value=False,
                visible=False,
            )
            create_sample_btn = gr.Button(
                t("generation.create_sample_btn"),
                variant="primary",
                size="lg",
                visible=False,
            )
        
        # State to track if sample has been created in Simple mode
        simple_sample_created = gr.State(value=False)
        
        # ==================== Source Audio (for Cover/Repaint) ====================
        # This is shown above the main content for Cover and Repaint modes
        with gr.Column(visible=False) as src_audio_group:
            with gr.Row(equal_height=True):
                # Source Audio - scale=10 to match (refer_audio=2 + prompt/lyrics=8)
                src_audio = gr.Audio(
                    label="Source Audio",
                    type="filepath",
                    scale=10,
                )
                # Process button - scale=1 to align with random button
                with gr.Column(scale=1, min_width=80):
                    process_src_btn = gr.Button(
                        "Analyze",
                        variant="secondary",
                        size="lg",
                    )
        
        # Hidden Audio Codes storage (needed internally but not displayed)
        text2music_audio_code_string = gr.Textbox(
            label="Audio Codes",
            visible=False,
        )
        
        # ==================== Custom/Cover/Repaint Mode Content ====================
        with gr.Column() as custom_mode_content:
            with gr.Row(equal_height=True):
                # Left: Reference Audio
                with gr.Column(scale=2, min_width=200):
                    reference_audio = gr.Audio(
                        label="Reference Audio (optional)",
                        type="filepath",
                        show_label=True,
                    )
                
                # Middle: Prompt + Lyrics + Format button
                with gr.Column(scale=8):
                    # Row 1: Prompt and Lyrics
                    with gr.Row(equal_height=True):
                        captions = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the music style, mood, instruments...",
                            lines=12,
                            max_lines=12,
                            scale=1,
                        )
                        lyrics = gr.Textbox(
                            label="Lyrics",
                            placeholder="Enter lyrics here... Use [Verse], [Chorus] etc. for structure",
                            lines=12,
                            max_lines=12,
                            scale=1,
                        )
                    
                    # Row 2: Format button (only below Prompt and Lyrics)
                    format_btn = gr.Button(
                        "Format",
                        variant="secondary",
                    )
                
                # Right: Random button
                with gr.Column(scale=1, min_width=60):
                    sample_btn = gr.Button(
                        "🎲",
                        variant="primary",
                        size="lg",
                    )
        
        # Placeholder for removed audio_uploads_accordion (for compatibility)
        audio_uploads_accordion = gr.Column(visible=False)
        
        # Legacy cover_mode_group (hidden, for backward compatibility)
        cover_mode_group = gr.Column(visible=False)
        # Legacy convert button (hidden, for backward compatibility)
        convert_src_to_codes_btn = gr.Button("Convert to Codes", visible=False)
        
        # ==================== Repaint Mode: Source + Time Range ====================
        with gr.Column(visible=False) as repainting_group:
            with gr.Row():
                repainting_start = gr.Number(
                    label="Start (seconds)",
                    value=0.0,
                    step=0.1,
                    scale=1,
                )
                repainting_end = gr.Number(
                    label="End (seconds, -1 for end)",
                    value=-1,
                    minimum=-1,
                    step=0.1,
                    scale=1,
                )
        
        # ==================== Remix Mode: Controls ====================
        with gr.Column(visible=False) as remix_group:
            gr.HTML("<p style='color: #a0aec0; font-size: 0.85em; margin: 0 0 8px 0;'>Remix creates a new version of the source song with a different style. Lower strength = more creative departure from original, higher = closer to original.</p>")
            with gr.Row():
                remix_strength = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Remix Strength",
                    info="How closely to follow the source audio (0.1=very creative, 1.0=close to original)",
                    scale=3,
                )
                remix_style_prompt = gr.Textbox(
                    label="Remix Style (optional)",
                    placeholder="e.g., 'lo-fi hip hop', 'EDM remix', 'jazz cover'...",
                    value="",
                    info="Describe the target remix style",
                    scale=5,
                )
        
        # ==================== Optional Parameters ====================
        with gr.Accordion("⚙️ Optional Parameters", open=False, visible=False) as optional_params_accordion:
            pass

        # ==================== Advanced Settings ====================
        with gr.Accordion("🔧 Advanced Settings", open=False) as advanced_options_accordion:
            with gr.Row():
                bpm = gr.Number(
                    label="BPM (optional)",
                    value=0,
                    step=1,
                    info="leave empty for N/A",
                    scale=1,
                )
                key_scale = gr.Textbox(
                    label="Key Signature (optional)",
                    placeholder="Leave empty for N/A",
                    value="",
                    info="A-G, #/♭, major/minor",
                    scale=1,
                )
                time_signature = gr.Dropdown(
                    choices=["", "2", "3", "4"],
                    value="",
                    label="Time Signature (optional)",
                    allow_custom_value=True,
                    info="2/4, 3/4, 4/4...",
                    scale=1,
                )
                audio_duration = gr.Number(
                    label="Audio Duration (seconds)",
                    value=-1,
                    minimum=-1,
                    maximum=600.0,
                    step=1,
                    info="Use -1 for auto, or 10-600 seconds",
                    scale=1,
                )
                vocal_language = gr.Dropdown(
                    choices=VALID_LANGUAGES,
                    value="unknown",
                    label="Vocal Language",
                    allow_custom_value=True,
                    info="use `unknown` for instrumental",
                    scale=1,
                )
                batch_size_input = gr.Number(
                    label="batch size",
                    info="max 8",
                    value=2,
                    minimum=1,
                    maximum=8,
                    step=1,
                    scale=1,
                    interactive=False,
                )
                
            # Row 1: DiT Inference Steps, Seed, Audio Format
            with gr.Row():
                inference_steps = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=8,
                    step=1,
                    label="DiT Inference Steps",
                    info="Turbo: max 8, Base: max 200",
                )
                seed = gr.Textbox(
                    label="Seed",
                    value="-1",
                    info="Use comma-separated values for batches",
                )
                audio_format = gr.Dropdown(
                    choices=["mp3", "flac"],
                    value="mp3",
                    label="Audio Format",
                    info="Audio format for saved files",
                )
            
            # Row 2: Shift, Random Seed, Inference Method
            with gr.Row():
                shift = gr.Slider(
                    minimum=1.0,
                    maximum=5.0,
                    value=3.0,
                    step=0.1,
                    label="Shift",
                    info="Timestep shift factor for base models (range 1.0-5.0, default 3.0). Not effective for turbo models.",
                )
                random_seed_checkbox = gr.Checkbox(
                    label="Random Seed",
                    value=True,
                    info="Enable to auto-generate seeds",
                )
                infer_method = gr.Dropdown(
                    choices=["ode", "sde"],
                    value="ode",
                    label="Inference Method",
                    info="Diffusion inference method. ODE (Euler) is faster, SDE (stochastic) may produce different results.",
                )
            
            # Row 3: Custom Timesteps (full width)
            custom_timesteps = gr.Textbox(
                label="Custom Timesteps",
                placeholder="0.97,0.76,0.615,0.5,0.395,0.28,0.18,0.085,0",
                value="",
                info="Optional: comma-separated values from 1.0 to 0.0 (e.g., '0.97,0.76,0.615,0.5,0.395,0.28,0.18,0.085,0'). Overrides inference steps and shift.",
            )
            
            # Section: LM Generation Parameters
            gr.HTML("<h4>🎵 LM Generation Parameters</h4>")
            
            # Row 4: LM Temperature, LM CFG Scale, LM Top-K, LM Top-P
            with gr.Row():
                lm_temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.85,
                    step=0.05,
                    label="LM Temperature",
                    info="5Hz LM temperature (higher = more random)",
                )
                lm_cfg_scale = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    value=2.0,
                    step=0.1,
                    label="LM CFG Scale",
                    info="5Hz LM CFG (1.0 = no CFG)",
                )
                lm_top_k = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    label="LM Top-K",
                    info="Top-k (0 = disabled)",
                )
                lm_top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.01,
                    label="LM Top-P",
                    info="Top-p (1.0 = disabled)",
                )
            
            # Row 5: LM Negative Prompt (full width)
            lm_negative_prompt = gr.Textbox(
                label="LM Negative Prompt",
                value="NO USER INPUT",
                placeholder="Things to avoid in generation...",
                lines=2,
                info="Negative prompt (use when LM CFG Scale > 1.0)",
            )
            # audio_cover_strength remains hidden for now
            audio_cover_strength = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, visible=False)
        
        # Note: audio_duration, bpm, key_scale, time_signature are now visible in Optional Parameters
        # ==================== Generate Button Row ====================
        generate_btn_interactive = init_params.get('enable_generate', False) if service_pre_initialized else False
        with gr.Row(equal_height=True):
            # Left: Thinking and Instrumental checkboxes
            with gr.Column(scale=1, min_width=120):
                think_checkbox = gr.Checkbox(
                    label="Thinking",
                    value=True,
                )
                instrumental_checkbox = gr.Checkbox(
                    label="Instrumental",
                    value=False,
                )
            
            # Center: Generate button
            with gr.Column(scale=4):
                generate_btn = gr.Button(
                    "🎵 Generate Music",
                    variant="primary",
                    size="lg",
                    interactive=generate_btn_interactive,
                )
            
            # Right: auto_score, auto_lrc
            with gr.Column(scale=1, min_width=120):
                auto_score = gr.Checkbox(
                    label="Get Scores",
                    value=False,
                )
                auto_lrc = gr.Checkbox(
                    label="Get LRC",
                    value=False,
                )
        
        # ==================== Hidden Components (for internal use) ====================
        # These are needed for event handlers but not shown in UI
        
        # Task type (set automatically based on generation_mode)
        actual_model = init_params.get('config_path', 'acestep-v15-turbo') if service_pre_initialized else 'acestep-v15-turbo'
        actual_model_lower = (actual_model or "").lower()
        if "turbo" in actual_model_lower:
            initial_task_choices = TASK_TYPES_TURBO
        else:
            initial_task_choices = TASK_TYPES_BASE
        
        task_type = gr.Dropdown(
            choices=initial_task_choices,
            value="text2music",
            visible=False,
        )
        
        instruction_display_gen = gr.Textbox(
            value=DEFAULT_DIT_INSTRUCTION,
            visible=False,
        )
        
        track_name = gr.Dropdown(
            choices=TRACK_NAMES,
            value=None,
            visible=False,
        )
        
        complete_track_classes = gr.CheckboxGroup(
            choices=TRACK_NAMES,
            visible=False,
        )
        
        # Note: lyrics, vocal_language, instrumental_checkbox, format_btn are now visible in custom_mode_content
        
        # Hidden advanced settings (keep defaults)
        # Note: Most parameters are now visible in Advanced Settings section above
        guidance_scale = gr.Slider(value=7.0, visible=False)
        use_adg = gr.Checkbox(value=False, visible=False)
        cfg_interval_start = gr.Slider(value=0.0, visible=False)
        cfg_interval_end = gr.Slider(value=1.0, visible=False)
        
        # LM parameters (remaining hidden ones)
        use_cot_metas = gr.Checkbox(value=True, visible=False)
        use_cot_caption = gr.Checkbox(value=True, visible=False)
        use_cot_language = gr.Checkbox(value=True, visible=False)
        constrained_decoding_debug = gr.Checkbox(value=False, visible=False)
        allow_lm_batch = gr.Checkbox(value=True, visible=False)
        lm_batch_chunk_size = gr.Number(value=8, visible=False)
        score_scale = gr.Slider(minimum=0.01, maximum=1.0, value=0.5, visible=False)
        autogen_checkbox = gr.Checkbox(value=False, visible=False)
        
        # Transcribe button (hidden)
        transcribe_btn = gr.Button(value="Transcribe", visible=False)
        text2music_audio_codes_group = gr.Group(visible=False)
        
        # Note: format_btn is now visible in custom_mode_content
        
        # Load file button (hidden for now)
        load_file = gr.UploadButton(
            label="Load",
            file_types=[".json"],
            file_count="single",
            visible=False,
        )
        
        # Caption/Lyrics accordions (not used in new UI but needed for compatibility)
        caption_accordion = gr.Accordion("Caption", visible=False)
        lyrics_accordion = gr.Accordion("Lyrics", visible=False)
        # Note: optional_params_accordion is now visible above
    
    return {
        "service_config_accordion": service_config_accordion,
        "language_dropdown": language_dropdown,
        "checkpoint_dropdown": checkpoint_dropdown,
        "refresh_btn": refresh_btn,
        "config_path": config_path,
        "device": device,
        "init_btn": init_btn,
        "init_status": init_status,
        "lm_model_path": lm_model_path,
        "init_llm_checkbox": init_llm_checkbox,
        "backend_dropdown": backend_dropdown,
        "use_flash_attention_checkbox": use_flash_attention_checkbox,
        "offload_to_cpu_checkbox": offload_to_cpu_checkbox,
        "offload_dit_to_cpu_checkbox": offload_dit_to_cpu_checkbox,
        # LoRA components
        "lora_path": lora_path,
        "load_lora_btn": load_lora_btn,
        "unload_lora_btn": unload_lora_btn,
        "use_lora_checkbox": use_lora_checkbox,
        "lora_status": lora_status,
        # DiT model selector
        "dit_model_selector": dit_model_selector,
        "task_type": task_type,
        "instruction_display_gen": instruction_display_gen,
        "track_name": track_name,
        "complete_track_classes": complete_track_classes,
        "audio_uploads_accordion": audio_uploads_accordion,
        "reference_audio": reference_audio,
        "src_audio": src_audio,
        "convert_src_to_codes_btn": convert_src_to_codes_btn,
        "text2music_audio_code_string": text2music_audio_code_string,
        "transcribe_btn": transcribe_btn,
        "text2music_audio_codes_group": text2music_audio_codes_group,
        "lm_temperature": lm_temperature,
        "lm_cfg_scale": lm_cfg_scale,
        "lm_top_k": lm_top_k,
        "lm_top_p": lm_top_p,
        "lm_negative_prompt": lm_negative_prompt,
        "use_cot_metas": use_cot_metas,
        "use_cot_caption": use_cot_caption,
        "use_cot_language": use_cot_language,
        "repainting_group": repainting_group,
        "repainting_start": repainting_start,
        "repainting_end": repainting_end,
        "audio_cover_strength": audio_cover_strength,
        # Remix mode components
        "remix_group": remix_group,
        "remix_strength": remix_strength,
        "remix_style_prompt": remix_style_prompt,
        # Generation mode components
        "generation_mode": generation_mode,
        "simple_mode_group": simple_mode_group,
        "simple_query_input": simple_query_input,
        "random_desc_btn": random_desc_btn,
        "simple_instrumental_checkbox": simple_instrumental_checkbox,
        "simple_vocal_language": simple_vocal_language,
        "create_sample_btn": create_sample_btn,
        "simple_sample_created": simple_sample_created,
        "caption_accordion": caption_accordion,
        "lyrics_accordion": lyrics_accordion,
        "optional_params_accordion": optional_params_accordion,
        # Custom mode components
        "custom_mode_content": custom_mode_content,
        "cover_mode_group": cover_mode_group,
        # Source audio group for Cover/Repaint
        "src_audio_group": src_audio_group,
        "process_src_btn": process_src_btn,
        "advanced_options_accordion": advanced_options_accordion,
        # Existing components
        "captions": captions,
        "sample_btn": sample_btn,
        "load_file": load_file,
        "lyrics": lyrics,
        "vocal_language": vocal_language,
        "bpm": bpm,
        "key_scale": key_scale,
        "time_signature": time_signature,
        "audio_duration": audio_duration,
        "batch_size_input": batch_size_input,
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "random_seed_checkbox": random_seed_checkbox,
        "use_adg": use_adg,
        "cfg_interval_start": cfg_interval_start,
        "cfg_interval_end": cfg_interval_end,
        "shift": shift,
        "infer_method": infer_method,
        "custom_timesteps": custom_timesteps,
        "audio_format": audio_format,
        "think_checkbox": think_checkbox,
        "autogen_checkbox": autogen_checkbox,
        "generate_btn": generate_btn,
        "instrumental_checkbox": instrumental_checkbox,
        "format_btn": format_btn,
        "constrained_decoding_debug": constrained_decoding_debug,
        "score_scale": score_scale,
        "allow_lm_batch": allow_lm_batch,
        "auto_score": auto_score,
        "auto_lrc": auto_lrc,
        "lm_batch_chunk_size": lm_batch_chunk_size,
    }
