import transformers

# ====================================================
# tokenizer
# ====================================================
def get_tokenizer(tokenizer_path, tokenizer_params, transformers_config=None):
    if transformers_config is None:
        transformers_config = transformers.AutoConfig.from_pretrained(tokenizer_path)
    transformers_config.update(tokenizer_params)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        config=transformers_config,
    )

    return tokenizer
