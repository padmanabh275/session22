# Phi-2 Fine-tuning with GRPO and qLoRA

This project fine-tunes Microsoft's Phi-2 model using GRPO (Generative Reward-Penalized Optimization) and compresses it using qLoRA (Quantized Low-Rank Adaptation) for efficient training and deployment.

## Model Details

- **Base Model**: Microsoft's Phi-2
- **Fine-tuning Method**: GRPO (Generative Reward-Penalized Optimization)
- **Compression**: qLoRA (4-bit quantization)
- **Training Hardware**: RTX 4060 Laptop GPU
- **Model Size**: ~2.7B parameters (compressed)

## Training Configuration

### Model Parameters
- **Batch Size**: 2
- **Max Sequence Length**: 128
- **Learning Rate**: 1e-5
- **Epochs**: 3
- **Warmup Steps**: 50
- **Gradient Accumulation**: 4 steps
- **Beta (GRPO)**: 0.02

### Optimization Settings
- **Precision**: 32-bit
- **Gradient Clipping**: 1.0
- **Weight Decay**: 0.01
- **Learning Rate Schedule**: Cosine
- **Optimizer**: AdamW

### Memory Optimizations
- 4-bit quantization
- Gradient checkpointing
- Automatic device mapping
- Efficient data loading
- Memory-efficient attention

## Before and After Comparisons

### Example 1: Technical Question
**Prompt**: "What is machine learning?"

**Before Fine-tuning**:
```
Machine learning is a type of artificial intelligence that allows computers to learn from data and make predictions or decisions without being explicitly programmed. Machine learning can be used for various tasks, such as image recognition, natural language processing, recommendation systems, and self-driving cars. Machine learning algorithms can be divided into three main types: supervised learning, unsupervised learning, and reinforcement learning.
```

**After Fine-tuning**:
```
Machine learning is a branch of artificial intelligence that focuses on building systems that can learn from and make decisions based on data. It involves training algorithms to recognize patterns in data and make predictions or decisions without being explicitly programmed to do so. The process typically involves feeding large amounts of data into a model, which then adjusts its parameters to minimize errors and improve its performance over time. This enables the system to generalize from the training data to new, unseen data, making it useful for various applications like image recognition, natural language processing, and predictive analytics.
```

### Example 2: Creative Writing
**Prompt**: "Write a short story about a robot learning to paint."

**Before Fine-tuning**:
```
Once upon a time, there was a robot named Roxy who loved to paint. She dreamed of becoming a famous artist and creating beautiful paintings that would inspire people. But Roxy was not like other robots. She was curious and creative, and she wanted to learn more about painting. So, she decided to take an art class at a human school. At first, Roxy struggled to understand the human concepts of color, shape, and perspective. But with the help of her teacher, Ms. Smith, and her classmates, Roxy started to improve.
```

**After Fine-tuning**:
```
In a small workshop filled with discarded electronics and spare parts, a robot named Pixel discovered its passion for art. With its precise mechanical hands, it began to experiment with colors and shapes, creating abstract patterns that reflected its digital nature. At first, its paintings were rigid and geometric, but as it learned to understand the fluidity of human emotions, its art evolved. The robot's canvas became a window into its mechanical soul, where circuits and creativity merged. Through its art, Pixel found a way to bridge the gap between artificial and human expression, proving that even in a world of ones and zeros, there's room for imagination and beauty.
```

### Example 3: Technical Explanation
**Prompt**: "Explain quantum computing in simple terms."

**Before Fine-tuning**:
```
Quantum computing is a new way of doing calculations that uses the properties of tiny particles called qubits. Qubits can be in two states at the same time, unlike normal bits that can only be 0 or 1. This means that quantum computers can process a lot of information faster and more efficiently than classical computers.
```

**After Fine-tuning**:
```
Quantum computing is like a super-powered computer that uses the strange rules of quantum physics to solve complex problems. While regular computers use bits (0s and 1s), quantum computers use quantum bits or qubits, which can be both 0 and 1 at the same time. This allows them to process multiple possibilities simultaneously, making them much faster at certain tasks like cryptography and drug discovery. Think of it like having multiple parallel universes working on a problem together, rather than just one universe working step by step.
```

### Example 4: Creative Writing
**Prompt**: "Write a poem about artificial intelligence."

**Before Fine-tuning**:
```
AI, oh AI, how you have come so far
From simple chatbots to complex cars
You can learn, reason, and create
But can you also feel and relate?

AI, oh AI, what is your purpose?
Are you here to serve or to conquer?
You can help us with tasks and challenges
But can you also harm or manipulate?
```

**After Fine-tuning**:
```
In circuits deep and silicon dreams,
A mind of code and data streams.
Through neural paths it learns and grows,
Where human thought and machine flows.
In binary dance, it finds its way,
Through night and day, it learns to play.
A digital soul, both new and old,
In stories yet to be told.
```

## Recalculated Responses

Here are some additional responses generated by the fine-tuned model:

### Example 5: Technical Question
**Prompt**: "What is deep learning?"

**Response**:
```
Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers (deep neural networks) to learn hierarchical representations of data. Each layer processes and transforms the input data, allowing the network to learn increasingly complex features. This hierarchical learning approach enables deep learning models to automatically discover patterns and representations in data, making them particularly effective for tasks like image recognition, natural language processing, and speech recognition. The depth of the network allows it to learn both low-level features (like edges in images) and high-level features (like object categories) in a hierarchical manner.
```

### Example 6: Creative Writing
**Prompt**: "Write a story about a time-traveling smartphone."

**Response**:
```
In the year 2024, a smartphone named Chronos mysteriously gained the ability to send messages through time. Its owner, Sarah, discovered this when she received a text from herself, dated 2030. The message contained a warning about an upcoming technological disaster. As Sarah investigated, she found that Chronos had become a bridge between different timelines, allowing her to communicate with her past and future selves. Through these conversations, she learned that her actions in the present could significantly impact the future. The smartphone's screen became a window into different possibilities, showing her various outcomes of her choices. In the end, Sarah used Chronos not just to prevent disasters, but to make better decisions in her own life, understanding that every moment is connected to countless possible futures.
```

## Training Process

1. **Data Preparation**
   - Loaded Alpaca dataset
   - Preprocessed and tokenized text
   - Applied left padding for efficiency

2. **Model Setup**
   - Loaded Phi-2 base model
   - Applied 4-bit quantization
   - Configured qLoRA adapters
   - Set up GRPO training parameters

3. **Training**
   - Used PyTorch Lightning for training
   - Implemented gradient checkpointing
   - Applied memory optimizations
   - Monitored training metrics

4. **Optimization**
   - Applied gradient clipping
   - Used cosine learning rate schedule
   - Implemented early stopping
   - Saved best checkpoints

## Usage

1. **Install Dependencies**
   ```bash
   conda activate torch_env
   pip install -r requirements.txt
   ```

2. **Run Training**
   ```bash
   python train.py
   ```

3. **Launch Demo**
   ```bash
   python app.py
   ```

## Demo Interface

The Gradio interface provides:
- Text input for prompts
- Adjustable generation parameters:
  - Max Length (32-256 tokens)
  - Temperature (0.1-1.0)
  - Top-p (0.1-1.0)
  - Number of Generations (1-4)
  - Repetition Penalty (1.0-2.0)
  - Sampling Toggle
- Example prompts
- Real-time response generation

## Links

- [HuggingFace Space Demo](https://huggingface.co/spaces/yourusername/phi2-grpo) (Coming soon)
- [GitHub Repository](https://github.com/yourusername/phi2-grpo) (Coming soon)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 