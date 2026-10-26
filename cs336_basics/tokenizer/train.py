import os
from typing import BinaryIO
import regex as re

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert all(isinstance(item, bytes) for item in split_special_token), "All elements in the list must be bytestrings"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    # 这里是整除
    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        file.seek(chunk_boundaries[bi]) 
        overleap = len(max(split_special_token, key=len)) - 1  # 特殊token中最长的
        tail = b""  # tail：保存上一次读取末尾的几个字节，防止特殊 token 被截断
        while True:
            data = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if not data:
                chunk_boundaries[bi] = file_size
                break
            mini_chunk = tail + data  # Prepend the tail from the previous mini chunk
            # Find the special token in the mini chunk
            found_at = min((pos for token in split_special_token if (pos := mini_chunk.find(token)) != -1), default=-1)
            if found_at != -1:
                chunk_boundaries[bi] = file.tell() - len(mini_chunk) + found_at # 计算绝对位置
                break
            tail = mini_chunk[-overleap:] if overleap else b""  

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
# 统计count中词语数量
def count_chunk_vocab(chunk: str, PAT: str):
    matches = re.finditer(PAT, chunk)
    
    vocab_count = {}
    for vocabulary in matches:
        if vocabulary.group() not in vocab_count:
            vocab_count[vocabulary.group()] = 0
        vocab_count[vocabulary.group()] += 1
    return vocab_count
def merge_chunk_counts(all_chunk_vocab_count: list):
    pre_tokenization_vocab_count = {}
    for chunk in all_chunk_vocab_count:
        for vocab, val in chunk.items():
            if vocab not in pre_tokenization_vocab_count:
                pre_tokenization_vocab_count[vocab] = val
            else:
                pre_tokenization_vocab_count[vocab] += val
    return pre_tokenization_vocab_count
def init_per_vocab(pre_token:dict):
    per_vocab = {}
    for key in pre_token:
        per_vocab[key] = [bytes([b]) for b in key.encode('utf-8')]
    return per_vocab

def merges_token(per_vocab: dict, pre_token: dict):
    pairs = {}
    # create pairs
    # piars: {"ab":1}
    # per_vocab: {"abc": ['a', 'b', 'c']} 合并之后会变化
    # pre_token: {"some": 3}
    # pairs: {("ab"): 3,....} 每个词中pair出现频率*词频率
    for vocab in per_vocab:
        tokens = per_vocab[vocab]
        if len(tokens) == 1:
            continue
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            if pair not in pairs:
                pairs[pair] = 0
            pairs[pair] += pre_token[vocab]
    if not pairs:
        return per_vocab, None
    # max_pair是字典
    max_pair = max(pairs, key=lambda pair: (pairs[pair], pair))
    # update per_vocab
    for vocab in per_vocab:
        tokens = per_vocab[vocab]
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == max_pair:
                new_tokens.append(tokens[i]+tokens[i+1])
                i+=2
            else:
                new_tokens.append(tokens[i])
                i+=1

        per_vocab[vocab] = new_tokens
    return per_vocab, max_pair
# special_token = b"<|endoftext|>"  # Define the special token as a bytestring
def train_bpe(input_path:str, vocab_size:int, special_token: list[str]):
    vocab = {} 
    merges = []
    # pertokenization rule
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


    num_processes = 4
    
    all_chunk_vocab_count = [] #[{some:1},{xxx,:2}]
    special_token_bytes = [token.encode('utf-8')for token in special_token]
    if vocab_size < 256 + len(special_token_bytes):
        raise ValueError("vocab_size must be >= 257")
    if not special_token:
        raise ValueError("special_token cannot be empty")
    # vocabulary initialization
    token_num = 0

    for _ in range(256):
        vocab[token_num] = bytes([token_num])    
        token_num += 1
    for per_special_bytes in special_token_bytes:
        vocab.update({token_num: per_special_bytes})
        token_num += 1
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_token_bytes)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start)
            # remove特殊token
            pattern = b"|".join(re.escape(token) for token in sorted(special_token_bytes, key=len, reverse=True))
            parts = re.split(pattern, chunk)
            for part in parts:
                if not part:
                    continue
                all_chunk_vocab_count.append(count_chunk_vocab(part.decode("utf-8", errors= "ignore"), PAT))
    pre_tokenization_count = merge_chunk_counts(all_chunk_vocab_count)

    # merge
    per_vocab = init_per_vocab(pre_token=pre_tokenization_count) # init
    while len(vocab) < vocab_size:

        per_vocab,max_pair = merges_token(per_vocab=per_vocab, pre_token=pre_tokenization_count)
        if  max_pair is None:
            break
        new_token = max_pair[0] + max_pair[1]
        vocab[token_num] = new_token
        merges.append(max_pair)
        token_num += 1
        
    return vocab, merges

    