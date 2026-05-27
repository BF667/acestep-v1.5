"""
Gradio UI Components Module
Contains all Gradio interface component definitions and layouts
"""
import gradio as gr
from acestep.gradio_ui.i18n import get_i18n, t
from acestep.gradio_ui.interfaces.dataset import create_dataset_section
from acestep.gradio_ui.interfaces.generation import create_generation_section
from acestep.gradio_ui.interfaces.result import create_results_section
from acestep.gradio_ui.interfaces.training import create_training_section
from acestep.gradio_ui.events import setup_event_handlers, setup_training_event_handlers


def create_gradio_interface(dit_handler, llm_handler, dataset_handler, init_params=None, language='en') -> gr.Blocks:
    """
    Create Gradio interface
    
    Args:
        dit_handler: DiT handler instance
        llm_handler: LM handler instance
        dataset_handler: Dataset handler instance
        init_params: Dictionary containing initialization parameters and state.
                    If None, service will not be pre-initialized.
        language: UI language code ('en', 'zh', 'ja', default: 'en')
        
    Returns:
        Gradio Blocks instance
    """
    # Initialize i18n with selected language
    i18n = get_i18n(language)
    
    with gr.Blocks(
        title=t("app.title"),
        theme=gr.themes.Soft(),
        css="""
        .main-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .section-header {
            background: linear-gradient(90deg, #4CAF50, #45a049);
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .lm-hints-row {
            align-items: stretch;
        }
        .lm-hints-col {
            display: flex;
        }
        .lm-hints-col > div {
            flex: 1;
            display: flex;
        }
        .lm-hints-btn button {
            height: 100%;
            width: 100%;
        }
        """
    ) as demo:
        
        gr.HTML(f"""
        <div class="main-header">
            <h1>{t("app.title")}</h1>
            <p>{t("app.subtitle")}</p>
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 10px 20px; border-radius: 8px; text-align: center; margin: 8px auto; max-width: 600px;">
                <span style="color: white; font-size: 15px;">
                    🚀 Want faster &amp; more stable experience? Try 
                    <a href="https://acemusic.ai" target="_blank" style="color: #ffd700; font-weight: bold; text-decoration: underline;">acemusic.ai</a>
                    — 100% free!
                </span>
            </div>
            <p style="margin-top: 0.5rem;">
                <a href="https://ace-step.github.io/ace-step-v1.5.github.io/" target="_blank">Project</a> |
                <a href="https://huggingface.co/collections/ACE-Step/ace-step-15" target="_blank">Hugging Face</a> |
                <a href="https://modelscope.cn/models/ACE-Step/ACE-Step-v1-5" target="_blank">ModelScope</a> |
                <a href="https://github.com/ACE-Step/ACE-Step-1.5" target="_blank">GitHub</a> |
                <a href="https://discord.gg/PeWDxrkdj7" target="_blank">Discord</a> |
                <a href="https://arxiv.org/abs/2602.00744" target="_blank">Technical Report</a>
            </p>
        </div>
        """)
        
        # Dataset Explorer Section
        dataset_section = create_dataset_section(dataset_handler)
        
        # Generation Section (pass init_params and language to support pre-initialization)
        generation_section = create_generation_section(dit_handler, llm_handler, init_params=init_params, language=language)
        
        # Results Section
        results_section = create_results_section(dit_handler)
        
        # Training Section (LoRA training and dataset builder)
        # Pass init_params to support hiding in service mode
        training_section = create_training_section(dit_handler, llm_handler, init_params=init_params)
        
        # Connect event handlers (pass init_params for multi-model support)
        setup_event_handlers(demo, dit_handler, llm_handler, dataset_handler, dataset_section, generation_section, results_section, init_params=init_params)
        
        # Connect training event handlers
        setup_training_event_handlers(demo, dit_handler, llm_handler, training_section)
    
    return demo
