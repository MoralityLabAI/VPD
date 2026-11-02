"""Record the default-versus-raw decoding discrepancy without changing token IDs."""
import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from experiments.nanbeige_reasoning.answer_decoding import AnswerDecoder,DECODING_POLICY
from experiments.nanbeige_reasoning.check_llama_steering import read,sha,write


def main():
    root=Path('artifacts/nanbeige_reasoning');run=root/'q4_answer_context_v1'
    path=Path('models/Nanbeige4.2-3B-tokenizer')
    default=AutoTokenizer.from_pretrained(path,use_fast=False,trust_remote_code=True)
    raw=AnswerDecoder(path);rows=[]
    for row in read(run/'results.json'):
        with np.load(run/row['archive']) as z:
            ids=z['token_ids'].tolist();generated=z['generated'].tolist()
            original=default.decode(ids,skip_special_tokens=False)
            assert raw.decode(ids)==row['prompt_text']
            assert raw.decode(generated)==row['generated_text']
            rows.append(dict(condition=row['condition'],source_archive_sha256=sha(run/row['archive']),
                native_prompt_characters=len(row['prompt_text']),default_hf_prompt_characters=len(original),
                default_hf_prompt_equals_native=original==row['prompt_text'],raw_hf_prompt_equals_native=True,
                raw_hf_generation_equals_native=True))
    report=dict(status='decoder_presentation_difference_documented',
        context_verification_sha256=sha(run/'verification.json'),reporter_sha256=sha(Path(__file__)),
        default_hf_configuration=dict(add_prefix_space=default.add_prefix_space,legacy=default.legacy,
            clean_up_tokenization_spaces=default.clean_up_tokenization_spaces),
        raw_decoding_policy=DECODING_POLICY,conditions=rows,
        scope='Decoding-only presentation settings; saved/model-facing token IDs and tokenizer files are unchanged.')
    write(root/'q4_answer_decoder_spacing_verification.json',report)
    print(json.dumps(report))


if __name__=='__main__':main()
