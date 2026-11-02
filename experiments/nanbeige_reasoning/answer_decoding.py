"""Independent decoding of saved output IDs without HF's inserted prefix spaces.

This adapter exposes decoding only. It never changes model-facing tokenization,
question IDs or the pinned tokenizer files.
"""
DECODING_POLICY=dict(
    implementation='HF slow LlamaTokenizer, decoding only',
    add_prefix_space=False,spaces_between_special_tokens=False,
    clean_up_tokenization_spaces=False,skip_special_tokens=False,
    reason='Render token content without spaces inserted by the HF presentation defaults.',
)


class AnswerDecoder:
    def __init__(self,path):
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(path,use_fast=False,trust_remote_code=True)
        assert type(self.tokenizer).__name__=='LlamaTokenizer' and self.tokenizer.legacy is False
        self.tokenizer.add_prefix_space=False

    def decode(self,tokens,*,skip_special_tokens=False):
        assert skip_special_tokens is False
        return self.tokenizer.decode(tokens,skip_special_tokens=False,
            spaces_between_special_tokens=False,clean_up_tokenization_spaces=False)
