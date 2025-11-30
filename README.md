# Sleep-Based Learning: Continual Memory Consolidation in LLMs

This repository contains the implementation of "Sleep-Based Learning," a bio-inspired architecture that mimics biological REM cycles to achieve continual learning in Large Language Models.
https://gallahat.substack.com/p/llm-sleep-based-learning-implementing


## Features

- **Wake State**: Standard chatbot interaction.
- **Sleep State**: Generates synthetic future trajectories ("dreams") based on recent interactions using a specialized "Dream Generator" model.
- **Memory Consolidation**: Fine-tunes the model on a mixed batch of new dreams, recalled memories (old dreams), and grounding data to prevent catastrophic forgetting.
- **Bio-Inspired**: Mimics the "Dream to Predict" hypothesis of biological REM sleep.

## Repository Structure

- `FullPipeline.ipynb`: The main notebook containing the full Sleep-Dream-Wake cycle implementation, including the Gradio interface.
- `DreamsGenTrain.ipynb`: Notebook for training the "Dream Generator" model.
- `grounding-dataset.json`: Dataset of general facts used for grounding during the sleep cycle.

## Installation (python 3.11.14)

1.  Clone the repository:
    ```bash
    git clone https://github.com/Gal-Lahat/sleep-based-learning.git
    cd sleep-based-learning
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Set up your Hugging Face token:
    - Create a `.env` file in the root directory.
    - Add your token: `HF_TOKEN=your_huggingface_token_here`

## Usage

### 1. Training the Dream Generator (optional)
Open `DreamsGenTrain.ipynb` to train the specialized model responsible for generating synthetic dreams. This notebook demonstrates how to fine-tune a model (e.g., Gemma-3 4B) to predict future conversation turns.

### 2. Running the Full Pipeline
Open `FullPipeline.ipynb` to run the interactive chatbot.
- Run all cells to initialize the model and the Gradio interface.
- Chat with the model.
- Type `/sleep` to trigger the Sleep-Dream-Wake cycle, where the model will generate dreams, train on them, and consolidate memories.

## How it Works

1.  **Interaction**: The user chats with the model.
2.  **Sleep Trigger**: When `/sleep` is invoked, the current conversation history is processed.
3.  **Dream Generation**: The Dream Generator creates synthetic future scenarios based on the conversation.
4.  **Mixed-Batch Training**: These new dreams are mixed with old dreams (recalled from storage) and grounding facts.
5.  **Fine-Tuning**: A lightweight LoRA adapter is trained on this mixed dataset.
6.  **Merging**: The adapter is merged into the base model (or a continuous learning adapter), updating the model's long-term memory.

## Citation

If you use this code or concept, please cite:

> Gal Lahat. (2025). *LLM Sleep Based Learning: Implementing REM-style Cycles and Synthetic Dreaming for Continual Memory Consolidation in LLMs*.
https://gallahat.substack.com/p/llm-sleep-based-learning-implementing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
