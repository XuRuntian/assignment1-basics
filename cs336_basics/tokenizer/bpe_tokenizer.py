from collections.abc import Iterable, Iterator
# from cs336_basics.tokenizer.train import *
import json
import re
class BPETokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_token: list[str] | None =None):
        self.vocab = vocab
        self.merges = merges
        self.spcial_token = special_token
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None =None):
        # 存在问题还需修改
        vocab = {}
        merges = []
        with open(vocab_filepath,'rb') as f:
            content = json.load(f)

            if special_tokens is not None:
                for token in special_tokens:
                    if token not in content:
                        content[token] = len(content)
            
            for k, v in content.items():
                k = k.encode('utf-8')
                vocab[v] = k
            
        with open(merges_filepath, 'rb') as f:
            for line in f:
                token = line.split()
                merges.append((token[0], token[1]))

        return cls(vocab, merges, special_tokens)
            
    def encode(self, text: str) -> list[int]:
        text_split_vocabulary_bytes = []
        pattern = "|".join(re.escape(token) for token in sorted(self.spcial_token, key=len, reverse=True))
        parts = re.split(pattern, text)
        for part in parts:
            matches = re.finditer(self.PAT, part)
            text_split_vocabulary_bytes.extend(vocabulary.group() for vocabulary in matches)
        
        

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass

