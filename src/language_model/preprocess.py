class Preprocessor(object):
    def __init__(self, tokenizer, max_length) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sep_token = self.tokenizer.sep_token

    def original_text(self, text):
        return text

    def convert_paragraph_split_to_sep(self, text):
        return text.replace("\r\n", "\n").replace("\n\n", f" {self.sep_token} ")
