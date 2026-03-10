from transformers import AutoTokenizer

llama_instruct = "meta-llama/Llama-3.1-8B-Instruct"
apertus_instruct = "swiss-ai/Apertus-8B-Instruct-2509"
mistral_instruct = "mistralai/Mistral-7B-Instruct-v0.3"
gpt_oss = "openai/gpt-oss-120b"
# Load the Apertus tokenizer
tokenizer = AutoTokenizer.from_pretrained(gpt_oss)

# Print basic special tokens
print("=== Basic Special Tokens ===")
print(f"BOS token: {tokenizer.bos_token!r} (ID: {tokenizer.bos_token_id})")
print(f"EOS token: {tokenizer.eos_token!r} (ID: {tokenizer.eos_token_id})")
print(f"PAD token: {tokenizer.pad_token!r} (ID: {tokenizer.pad_token_id})")

# Print all special tokens
print("\n=== All Special Tokens ===")
for token_name, token_value in tokenizer.special_tokens_map.items():
    if isinstance(token_value, str):
        token_id = tokenizer.convert_tokens_to_ids(token_value)
        print(f"{token_name}: {token_value!r} (ID: {token_id})")


# Print the full vocabulary of special tokens
print("\n=== Additional Special Tokens ===")
if hasattr(tokenizer, "added_tokens_encoder"):
    for token, token_id in tokenizer.added_tokens_encoder.items():
        print(f"{token!r} (ID: {token_id})")

# Apply chat template with sample messages
print("\n=== Chat Template Example ===")
messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What is its population?"},
]

# Apply chat template
formatted_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("\nFormatted chat:")
print(formatted_chat)

# Tokenize and print token IDs
tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
print(f"\nToken IDs: {tokens}")

# Decode tokens to show the mapping
print("\n=== Token-by-Token Breakdown ===")
for i, token_id in enumerate(tokens):
    token_str = tokenizer.decode([token_id])
    print(f"Position {i:3d}: ID {token_id:6d} -> {token_str!r}")
