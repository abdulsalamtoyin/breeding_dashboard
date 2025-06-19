
# Using Local LLMs with Breeding Dashboard

## Option 1: Ollama (Recommended for beginners)

### Installation:
1. Install Ollama: https://ollama.ai/
2. Run: `ollama pull llama2:7b` or `ollama pull mistral:7b`
3. Start server: `ollama serve`

### Configuration:
- Set USE_LOCAL_LLM=true in .env
- Set LOCAL_LLM_URL=http://localhost:11434

## Option 2: Transformers with Hugging Face

### Models to consider:
- microsoft/DialoGPT-large
- microsoft/BioGPT-Large
- distilbert-base-uncased (for embeddings)

### Memory requirements:
- 7B models: ~14GB RAM
- 13B models: ~26GB RAM
- Consider quantized versions for lower memory

## Option 3: OpenAI-compatible APIs

### LocalAI:
- Docker-based solution
- OpenAI API compatible
- Supports multiple model formats

### Considerations:
- Local models may have lower accuracy than GPT-3.5/4
- Significantly slower on CPU
- No internet required (privacy benefit)
- One-time setup cost vs. per-token pricing
