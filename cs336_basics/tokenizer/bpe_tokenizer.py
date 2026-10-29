from collections.abc import Iterable, Iterator
# from cs336_basics.tokenizer.train import *
import json
class BPETokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_token: list[str] | None =None):
        self.vocab = vocab
        self.merges = merges
        self.spcial_token = special_token
        
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
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass

