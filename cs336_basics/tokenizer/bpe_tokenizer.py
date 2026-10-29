from collections.abc import Iterable, Iterator
# from cs336_basics.tokenizer.train import *
import json
import regex as re
class BPETokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_token: list[str] | None =None):
        self.vocab = vocab
        self.merges = merges
        self.spcial_token = special_token
        self.vocab_trun = {v: k for k, v in self.vocab.items()}
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
        merged_vocabulary = []
        if self.spcial_token:
            pattern = "|".join(re.escape(token) for token in sorted(self.spcial_token, key=len, reverse=True))
            parts = re.split(pattern, text)
        else:
            parts = text
        ids = []
        # 分割为每个词语
        for part in parts:
            matches = re.finditer(self.PAT, part)
            text_split_vocabulary_bytes.extend(vocabulary.group().encode('utf-8') for vocabulary in matches)
        # merge
        for vocabulary in text_split_vocabulary_bytes:
            i = 0
            pre_char = b""
            while i < len(vocabulary) - 1:
                pair = (pre_char + vocabulary[i], vocabulary[i+1]) 
                if pair in self.merges:
                    pre_char += vocabulary[i]
                else:
                    merged_vocabulary.append(pre_char+vocabulary[i:i+1])
                    pre_char = b""
                i+=1
            merged_vocabulary.append(pre_char+vocabulary[i:i+1])
        # token化
        for vocabulary in merged_vocabulary:
            if vocabulary in self.vocab_trun:
                ids.append(self.vocab_trun[vocabulary])
            else:
                ids.append("u+FFFD")#这里如果encode词表没有得映射

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for iter in iterable:
            return self.encode(iter)


    def decode(self, ids: list[int]) -> str:
        text = ""
        for id in ids:
            text += self.vocab[id].decode('utf-8')

        return text
