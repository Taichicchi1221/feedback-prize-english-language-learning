class Preprocessor(object):
    def __init__(self, tokenizer, max_length) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sep_token = self.tokenizer.sep_token

    def raw_text(self, text):
        return text

    def convert_paragraph_split_to_sep(self, text):
        return text.replace("\r\n", "\n").replace("\n\n", f" {self.sep_token} ")

    def first_paragraph(self, text):
        paragraphs = text.replace("\r\n", "\n").split("\n\n")
        return paragraphs[0]

    def last_paragraph(self, text):
        paragraphs = text.replace("\r\n", "\n").split("\n\n")
        return paragraphs[-1]

    def join_first_last_paragraph(self, text):
        paragraphs = text.replace("\r\n", "\n").split("\n\n")
        if len(paragraphs) <= 1:
            return text
        return paragraphs[0] + f" {self.sep_token} " + paragraphs[-1]
