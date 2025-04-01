import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from torch.utils.data import Dataset, DataLoader

# Enable Tensor Core optimization for RTX GPUs
torch.set_float32_matmul_precision('medium')

# Initialize rich console for better logging
console = Console()

class TextDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Ensure tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Combine instruction and input if they exist
        prompt = item.get("instruction", "")
        if item.get("input"):
            prompt += "\n" + item["input"]
        
        # Tokenize the prompt
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "prompt": prompt
        }

class GRPOModel(pl.LightningModule):
    def __init__(
        self,
        model_name="microsoft/phi-2",
        learning_rate=2e-5,
        num_train_epochs=3,
        warmup_steps=100,
        batch_size=2,
        max_length=128,
        beta=0.04,
        num_generations=2,
        train_dataset=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Store train dataset
        self.train_dataset = train_dataset
        
        # Configure 4-bit quantization with memory optimizations
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=torch.float16,
        )
        
        # Load model with quantization and memory optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Store model name for reference model
        self.model_name = model_name
        self.ref_model = None

    def setup(self, stage=None):
        # Move model to the correct device after initialization
        if stage == "fit":
            self.model = self.model.to(self.device)

    def get_reference_model(self):
        if self.ref_model is None:
            # Load reference model with quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=None,
                trust_remote_code=True,
            )
            self.ref_model.eval()
            self.ref_model = self.ref_model.to(self.device)
        return self.ref_model

    def reward_function(self, completions):
        rewards = []
        for completion in completions:
            # Reward based on length (normalized)
            length_reward = len(completion.split()) / 100
            
            # Reward based on diversity (unique words)
            unique_words = len(set(completion.lower().split()))
            diversity_reward = unique_words / len(completion.split())
            
            # Combined reward
            reward = 0.7 * length_reward + 0.3 * diversity_reward
            rewards.append(reward)
        
        return torch.tensor(rewards, device=self.device)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.logits

    def training_step(self, batch, batch_idx):
        # Generate completions
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        prompts = batch["prompt"]
        
        # Generate multiple completions for each prompt
        all_completions = []
        for _ in range(self.hparams.num_generations):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            completions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_completions.extend(completions)
        
        # Calculate rewards
        rewards = self.reward_function(all_completions)
        
        # Calculate KL divergence
        ref_model = self.get_reference_model()
        with torch.no_grad():
            ref_outputs = ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            ref_logits = ref_outputs.logits
        
        policy_logits = self(input_ids, attention_mask)
        kl_div = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(policy_logits, dim=-1),
            torch.nn.functional.softmax(ref_logits, dim=-1),
            reduction='batchmean'
        )
        
        # Calculate GRPO loss
        loss = -rewards.mean() + self.hparams.beta * kl_div
        
        self.log("train_loss", loss)
        self.log("train_reward", rewards.mean())
        self.log("train_kl_div", kl_div)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.num_train_epochs * len(self.train_dataloader())
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "frequency": 1
            }
        }

    def on_train_end(self):
        # Clean up reference model to free memory
        if self.ref_model is not None:
            del self.ref_model
            self.ref_model = None
            torch.cuda.empty_cache()

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Train dataset not provided")
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True
        )

class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        max_length=256,
        batch_size=4,
        num_workers=4,
        pin_memory=True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

def main():
    # Load dataset
    dataset = load_dataset("tatsu-lab/alpaca")
    train_dataset = dataset["train"].select(range(500))
    
    # Initialize tokenizer with left padding
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # Create dataset with reduced max length
    train_dataset = TextDataset(train_dataset, tokenizer, max_length=128)
    
    # Initialize model with optimized parameters for RTX 4060 Laptop
    model = GRPOModel(
        train_dataset=train_dataset,
        batch_size=2,
        num_generations=2,
        max_length=128,
        learning_rate=1e-5,
        beta=0.02,
    )
    
    # Initialize logger and callbacks
    wandb_logger = WandbLogger(project="llm-finetuning")
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="model-{epoch:02d}-{step:04d}",
        monitor="train_loss",
        mode="min",
        save_top_k=3,
    )
    early_stopping = EarlyStopping(
        monitor="train_loss",
        patience=3,
        mode="min",
    )
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir="./fine-tuned-model",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        fp16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        report_to="wandb",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        torch_compile=True,
        max_grad_norm=1.0,
        group_by_length=True,
    )
    
    # Initialize trainer with memory-optimized settings
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="gpu",
        devices=1,
        precision="32",
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        log_every_n_steps=10,
        val_check_interval=0.5,
        callbacks=[
            checkpoint_callback,
            early_stopping,
        ],
        strategy="auto",
    )
    
    # Train the model
    console.print("[bold green]Starting training...[/bold green]")
    console.print("[bold yellow]Training with optimized settings for RTX 4060 Laptop GPU[/bold yellow]")
    console.print(f"Batch size: {model.hparams.batch_size}")
    console.print(f"Generations per prompt: {model.hparams.num_generations}")
    console.print(f"Max sequence length: {model.hparams.max_length}")
    trainer.fit(model)
    console.print("[bold green]Training completed![/bold green]")
    
    # Save the model
    model.model.save_pretrained("./fine-tuned-model")
    model.tokenizer.save_pretrained("./fine-tuned-model")
    console.print("[bold green]Model saved successfully![/bold green]")
    
    # Test the model
    test_prompt = "What is machine learning?"
    console.print("\n[bold blue]Testing the model:[/bold blue]")
    console.print(f"Original prompt: {test_prompt}")
    
    inputs = model.tokenizer(test_prompt, return_tensors="pt").to(model.device)
    outputs = model.model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    response = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    console.print(f"Generated response: {response}")

if __name__ == "__main__":
    main() 