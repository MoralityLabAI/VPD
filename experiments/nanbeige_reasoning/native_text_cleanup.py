"""Postprocessing-only reproduction of pinned llama.cpp detokenize cleanup.

This deliberately reproduces its three in-place byte passes, including reads
from the mutated buffer. It is not general whitespace normalization. Raw HF
decoding remains the primary scoring representation.
"""
SOURCE_REVISION = '0eadefebd3f8f92a86d634a0e5b8fffc9dc792c0'


def native_cleanup(text):
    data = bytearray(text.encode('utf-8'))
    # The pinned GGUF sets tokenizer.ggml.add_space_prefix=true. With
    # remove_special=False, detokenize strips at most one leading ASCII space.
    if data[:1] == b' ':
        del data[0]
    size = len(data)
    total = int(bool(size))
    for i in range(1, size):
        x = data[i]
        if data[i-1] == 32 and x in b'?!.,':
            total -= 1
        data[total] = x
        total += 1
    size = total
    total = int(bool(size))
    i = 1
    while i < size:
        x = data[i]
        if x == 39 and i+1 < size and data[i-1] == 32 and data[i+1] == 32:
            total -= 1
            i += 1
            data[i] = 0
        data[total] = x
        total += 1
        i += 1
    size = total
    total = int(bool(size))
    for i in range(1, size):
        x = data[i]
        if data[i-1] == 32 and x == 39 and i+1 < size:
            x1 = data[i+1]
            if x1 in b'sm':
                total -= 1
            elif x1 not in b'td' and i+2 < size:
                if bytes(data[i+1:i+3]) in (b're', b've'):
                    total -= 1
        data[total] = x
        total += 1
    return bytes(data[:total]).decode('utf-8', errors='replace')


def verify_native_text(raw, saved):
    if native_cleanup(raw) != saved:
        raise ValueError('Native text is not the exact pinned cleanup of raw HF decoding')
    return raw != saved


def verify_cleanup_receipt(path):
    from pathlib import Path
    from .check_llama_steering import read, sha
    path = Path(path)
    receipt = read(path)
    assert receipt['status'] == 'native_cleanup_independently_verified'
    assert receipt['source_revision'] == SOURCE_REVISION
    assert receipt['cleanup_sha256'] == sha(Path(__file__))
    assert receipt['raw_decoder_sha256'] == sha(Path(__file__).with_name('answer_decoding.py'))
    assert receipt['source_sha256'] == sha(Path('artifacts/nanbeige_reasoning/sources/llama_cpp/llama-vocab.cpp'))
    for name, digest in receipt['files'].items():
        asset = (path.parent/name).resolve()
        assert asset.parent == path.parent.resolve() and sha(asset) == digest
    return receipt
