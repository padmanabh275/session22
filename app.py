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
.prompt-box {
    background-color: #ffffff;
    border: 2px solid #3498db;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.prompt-box label {
    font-size: 1.1em;
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 10px;
    display: block;
}
.prompt-box textarea {
    width: 100%;
    min-height: 100px;
    padding: 10px;
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    font-size: 1em;
    line-height: 1.5;
}
.output-box {
    background-color: #ffffff;
    border: 2px solid #2ecc71;
    border-radius: 8px;
    padding: 20px;
    margin-top: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.output-box label {
    font-size: 1.1em;
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 15px;
    display: block;
}
.output-box .markdown {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 6px;
    border: 1px solid #e9ecef;
}
.output-box h3 {
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 8px;
    margin-top: 20px;
}
.output-box p {
    line-height: 1.6;
    color: #34495e;
    margin: 10px 0;
}
.loading {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    background-color: #f8f9fa;
    border-radius: 8px;
    margin: 10px 0;
}
.loading-spinner {
    width: 40px;
    height: 40px;
    border: 4px solid #f3f3f3;
    border-top: 4px solid #3498db;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-right: 15px;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.loading-text {
    color: #2c3e50;
    font-size: 1.1em;
    font-weight: 500;
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
            with gr.Column(elem_classes="prompt-box"):
                prompt = gr.Textbox(
                    label="Enter Your Prompt Here",
                    placeholder="Type your prompt here... (e.g., 'What is machine learning?' or 'Write a story about a robot learning to paint')",
                    lines=5,
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
            
            generate_btn = gr.Button("Generate", variant="primary", size="large")
        
        with gr.Column(scale=3):
            with gr.Column(elem_classes="output-box"):
                output = gr.Markdown(
                    label="Generated Response(s)",
                    show_label=True,
                    value="Your generated responses will appear here...",  # Add default value
                )
                loading_status = gr.Markdown(
                    value="",
                    show_label=False,
                    elem_classes="loading"
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
    
    def generate_with_status(*args):
        # Show loading status
        loading_status.value = """
        <div class="loading">
            <div class="loading-spinner"></div>
            <div class="loading-text">Generating responses... Please wait...</div>
        </div>
        """
        # Generate response
        result = generate_response(*args)
        # Clear loading status
        loading_status.value = ""
        return result
    
    # Connect the interface
    generate_btn.click(
        fn=generate_with_status,
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