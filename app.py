import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from rich.console import Console
import time

# Initialize rich console for better logging
console = Console()

# Load the model and tokenizer with the same configuration as training
console.print("[bold green]Loading model and tokenizer...[/bold green]")

# Configure 4-bit quantization with memory optimizations
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_quant_storage=torch.float16,
)

# Load model with quantization and memory optimizations
model = AutoModelForCausalLM.from_pretrained(
    "./fine-tuned-model",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-model")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

def generate_response(
    prompt,
    max_length=128,  # Match training max_length
    temperature=0.7,
    top_p=0.9,
    num_generations=2,  # Match training num_generations
    repetition_penalty=1.1,
    do_sample=True,
):
    try:
        # Get the device of the model
        device = next(model.parameters()).device
        
        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():  # Disable gradient computation
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_generations,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode and return the responses
        responses = []
        for output in outputs:
            response = tokenizer.decode(output, skip_special_tokens=True)
            responses.append(response)
        
        return "\n\n---\n\n".join(responses)
    except Exception as e:
        console.print(f"[bold red]Error during generation: {str(e)}[/bold red]")
        return f"Error: {str(e)}"

# Create custom CSS for better UI
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.container {
    max-width: 800px;
    margin: auto;
    padding: 20px;
}
.title {
    text-align: center;
    color: #2c3e50;
    margin-bottom: 20px;
}
.description {
    color: #34495e;
    line-height: 1.6;
    margin-bottom: 20px;
}
"""

# Create the Gradio interface with enhanced UI
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Phi-2 Fine-tuned with GRPO and qLoRA
        This model has been fine-tuned using GRPO (Generative Reward-Penalized Optimization) and compressed using qLoRA.
        Try it out with different prompts and generation parameters!
        """,
        elem_classes="title"
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                lines=3,
                show_label=True,
            )
            
            with gr.Row():
                with gr.Column():
                    max_length = gr.Slider(
                        minimum=32,
                        maximum=256,
                        value=128,
                        step=32,
                        label="Max Length",
                        info="Maximum number of tokens to generate"
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                        info="Higher values make output more random, lower values more deterministic"
                    )
                with gr.Column():
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.1,
                        label="Top-p",
                        info="Nucleus sampling parameter"
                    )
                    num_generations = gr.Slider(
                        minimum=1,
                        maximum=4,
                        value=2,
                        step=1,
                        label="Number of Generations",
                        info="Number of different responses to generate"
                    )
            
            with gr.Row():
                with gr.Column():
                    repetition_penalty = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.1,
                        step=0.1,
                        label="Repetition Penalty",
                        info="Higher values prevent repetition"
                    )
                with gr.Column():
                    do_sample = gr.Checkbox(
                        value=True,
                        label="Enable Sampling",
                        info="Enable/disable sampling for deterministic output"
                    )
            
            generate_btn = gr.Button("Generate", variant="primary")
        
        with gr.Column(scale=3):
            output = gr.Textbox(
                label="Generated Response(s)",
                lines=10,
                show_label=True,
            )
    
    gr.Markdown(
        """
        ### Example Prompts
        Try these example prompts to test the model:
        
        1. **Technical Question**: "What is machine learning?"
        2. **Creative Writing**: "Write a short story about a robot learning to paint."
        3. **Technical Explanation**: "Explain quantum computing in simple terms."
        4. **Creative Writing**: "Write a poem about artificial intelligence."
        """,
        elem_classes="description"
    )
    
    # Add examples
    gr.Examples(
        examples=[
            ["What is machine learning?"],
            ["Write a short story about a robot learning to paint."],
            ["Explain quantum computing in simple terms."],
            ["Write a poem about artificial intelligence."]
        ],
        inputs=prompt
    )
    
    # Connect the interface
    generate_btn.click(
        fn=generate_response,
        inputs=[
            prompt,
            max_length,
            temperature,
            top_p,
            num_generations,
            repetition_penalty,
            do_sample
        ],
        outputs=output
    )

if __name__ == "__main__":
    console.print("[bold green]Starting Gradio interface...[/bold green]")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True  # Enable sharing for HuggingFace Spaces
    ) 