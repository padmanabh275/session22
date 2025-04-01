import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console

console = Console()

def generate_responses():
    # Load the model and tokenizer
    console.print("[bold green]Loading model and tokenizer...[/bold green]")
    model = AutoModelForCausalLM.from_pretrained("./fine-tuned-model")
    tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-model")
    
    # Test prompts
    prompts = [
        "What is machine learning?",
        "Write a short story about a robot learning to paint.",
        "Explain quantum computing in simple terms.",
        "Write a poem about artificial intelligence."
    ]
    
    # Generate responses
    responses = {}
    for prompt in prompts:
        console.print(f"\n[bold blue]Generating response for: {prompt}[/bold blue]")
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses[prompt] = response
        console.print(f"[green]Response:[/green]\n{response}\n")
    
    # Save responses to a file
    with open("model_responses.txt", "w", encoding="utf-8") as f:
        for prompt, response in responses.items():
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Response: {response}\n")
            f.write("-" * 80 + "\n")
    
    console.print("[bold green]Responses saved to model_responses.txt[/bold green]")

if __name__ == "__main__":
    generate_responses() 