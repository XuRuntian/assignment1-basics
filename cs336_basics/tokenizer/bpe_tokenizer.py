from collections.abc import Iterable, Iterator
# from cs336_basics.tokenizer.train import *
import json
import regex as re
# PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_tokenizer.py::test_encode_iterable_tinystories_sample_roundtrip -v没通过
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
        text_split_vocabulary_bytes = [] # [[b'a',b'v'],[b'c']]
        merged_vocabulary = []
        if self.spcial_token:
            pattern = "("+ "|".join(re.escape(token) for token in sorted(self.spcial_token, key=len, reverse=True)) + ")" #正则表达组捕获
            parts = re.split(pattern, text)
        else:
            parts = [text]
            self.spcial_token = []
        ids = []
        # 分割为每个词语
        for part in parts:
            if part in self.spcial_token:
                text_split_vocabulary_bytes.extend([part.encode('utf-8')]) # 特殊token作为整体送入
                continue
            matches = re.finditer(self.PAT, part)
            text_split_vocabulary_bytes.extend(vocabulary.group().encode('utf-8') for vocabulary in matches)
                
        for i in range(len(text_split_vocabulary_bytes)):
            vocabulary = text_split_vocabulary_bytes[i] 
            if vocabulary.decode('utf-8') in self.spcial_token:
                text_split_vocabulary_bytes[i] = [vocabulary]
                continue
            text_split_vocabulary_bytes[i] = []
            for j in range(len(vocabulary)):
                text_split_vocabulary_bytes[i].append(vocabulary[j:j+1])
        
        merged_vocabulary = []
        for vocabulary in text_split_vocabulary_bytes:
            while len(vocabulary) > 1:
                pairs = []
                for i in range(len(vocabulary) - 1):
                    pairs.append((vocabulary[i], vocabulary[i+1]))
                # 合并
                for merge in self.merges:
                    if merge in pairs:
                        index = pairs.index(merge)
                        vocabulary[index:index+2] = [vocabulary[index] + vocabulary[index+1]] #合并为一个byte元素放入，而不是数值相加
                        break
                else:
                    break
            merged_vocabulary.extend(token for token in vocabulary)

        # token化
        for vocabulary in merged_vocabulary:
            if vocabulary in self.vocab_trun:
                ids.append(self.vocab_trun[vocabulary])
            else:
                print(f"遇到不存在的id对应")

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for iter in iterable:
            tokens =  self.encode(iter)
            for token in tokens:
                yield token


    def decode(self, ids: list[int]) -> str:
        text_encode = b""
        for id in ids:
            text_encode += self.vocab[id]

        return text_encode.decode('utf-8', errors="replace")
