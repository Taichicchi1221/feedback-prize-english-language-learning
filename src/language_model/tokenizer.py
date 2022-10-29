import transformers

# ====================================================
# tokenizer
# ====================================================
def get_tokenizer(tokenizer_path, tokenizer_params):
    if tokenizer_params is None:
        tokenizer_params = {}

    config = transformers.AutoConfig.from_pretrained(tokenizer_path)
    config.update(tokenizer_params)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, config=config)

    return tokenizer
