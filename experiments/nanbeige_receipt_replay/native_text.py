"""Explicit llama.cpp detokenization cleanup; retain the pre-cleanup text separately.

Port of pinned sources/llama_cpp/llama-vocab.cpp detokenize(), lines 3754-3814.
Byte-wise in-place passes mirror the native implementation, including contractions.
"""


def clean_native_spaces(text):
    data=bytearray(text.encode('utf-8'));total=len(data)
    for phase in range(3):
        previous=total;total=1 if total else 0;i=1
        while i<previous:
            x=data[i]
            if phase==0:
                if data[i-1]==32 and x in b'?!.,':total-=1
            elif phase==1:
                if x==39 and i+1<previous and data[i-1]==32 and data[i+1]==32:
                    total-=1;i+=1;data[i]=0
            elif data[i-1]==32 and x==39 and i+1<previous:
                x1=data[i+1]
                if x1 in b'sm':total-=1
                elif i+2<previous and bytes(data[i+1:i+3]) in (b're',b've'):total-=1
            data[total]=x;total+=1;i+=1
    return bytes(data[:total]).decode('utf-8')


def verify_native_text(tokenizer_text,native_text):
    if tokenizer_text==native_text:return 'exact'
    if clean_native_spaces(tokenizer_text)==native_text:return 'native_space_cleanup'
    raise ValueError('Native/tokenizer text differs beyond pinned space cleanup')
