import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from rich.console import Console
import time

# Initialize rich console for better logging
console = Console()

# Load the model and tokenizer with the same configuration as training
console.print("[bold green]Loading model and tokenizer...[/bold green]")

# Load model with memory optimizations
model = AutoModelForCausalLM.from_pretrained(
    "./fine-tuned-model",
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,  # Use float16 for memory efficiency
    low_cpu_mem_usage=True,  # Add this for better memory handling
)
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-model")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# Load base model for before/after comparison
console.print("[bold green]Loading base model for comparison...[/bold green]")
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,  # Add this for better memory handling
)

def generate_response(
    prompt,
    max_length=128,  # Match training max_length
    temperature=0.7,
    top_p=0.9,
    num_generations=2,  # Match training num_generations
    repetition_penalty=1.1,
    do_sample=True,
    show_comparison=True,  # New parameter for comparison toggle
):
    try:
        # Get the device of the model
        device = next(model.parameters()).device
        
        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response from fine-tuned model
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
        
        fine_tuned_response = "\n\n---\n\n".join(responses)
        
        if show_comparison:
            # Generate response from base model
            with torch.no_grad():
                base_outputs = base_model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=1,  # Only one for comparison
                    repetition_penalty=repetition_penalty,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
            
            return f"""
### Before Fine-tuning (Base Model)
{base_response}

### After Fine-tuning
{fine_tuned_response}
"""
        else:
            return fine_tuned_response
            
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
.comparison {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
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
            
            show_comparison = gr.Checkbox(
                value=True,
                label="Show Before/After Comparison",
                info="Toggle to show responses from both base and fine-tuned models"
            )
            
            generate_btn = gr.Button("Generate", variant="primary")
        
        with gr.Column(scale=3):
            output = gr.Markdown(
                label="Generated Response(s)",
                show_label=True,
            )
    
    gr.Markdown(
        """
        ### Example Prompts
        Try these example prompts to test the model:
        
        1. **Technical Questions**:
           - "What is machine learning?"
           - "What is deep learning?"
           - "What is the difference between supervised and unsupervised learning?"
        
        2. **Creative Writing**:
           - "Write a short story about a robot learning to paint."
           - "Write a story about a time-traveling smartphone."
           - "Write a fairy tale about a computer learning to dream."
           - "Create a story about an AI becoming an artist."
        
        3. **Technical Explanations**:
           - "How does neural network training work?"
           - "Explain quantum computing in simple terms."
           - "What is transfer learning?"
        
        4. **Creative Tasks**:
           - "Write a poem about artificial intelligence."
           - "Write a poem about the future of technology."
           - "Create a story about a robot learning to dream."
        """,
        elem_classes="description"
    )
    
    # Add examples
    gr.Examples(
        examples=[
            ["What is machine learning?"],
            ["Write a short story about a robot learning to paint."],
            ["Explain quantum computing in simple terms."],
            ["Write a poem about artificial intelligence."],
            ["What is deep learning?"],
            ["Write a story about a time-traveling smartphone."],
            ["How does neural network training work?"],
            ["Write a fairy tale about a computer learning to dream."],
            ["What is the difference between supervised and unsupervised learning?"],
            ["Create a story about an AI becoming an artist."]
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
            do_sample,
            show_comparison
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