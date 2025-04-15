# import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import os
from pathlib import Path
import regex as re
from .pretokenization_example import find_chunk_boundaries # for the test 
# from pretokenization_example import find_chunk_boundaries # for running bpe.py
import multiprocessing
from dataclasses import dataclass
from abc import ABC
from typing import List, Tuple, Dict, Iterable, Iterator
from tqdm import tqdm
import gc
import cProfile
import pstats
import io


PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: Dict[int, bytes]             # index -> bytes
    merges: Dict[Tuple[int, int], int]  # (index1, index2) -> new_index

class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, indices: List[int]) -> str:
        raise NotImplementedError


class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams, special_tokens=None):
        self.params = params
        self.special_tokens = special_tokens
        self.reverse_vocab = {v: k for k, v in self.params.vocab.items()} # find some way to do "is prefix"
        self.max_len_token = max([len(word) for word in self.params.vocab.values()]) # 128 

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        def encode_item_to_bytes(item):
            # For byte escape sequences (e.g., "\\xf8"), convert them to actual bytes
            RE = r"(\\x[0-9a-fA-F]{2})"
            item_list = [string for string in re.split(RE, item) if len(string)]
            b = b""
            for item in item_list:
                if item.startswith("\\x"):
                    b += bytes([int(item[2:], 16)])
                else:
                    b += item.encode("utf-8")
            return b

        with open(vocab_filepath, 'r') as file:
            vocab = {int(num): encode_item_to_bytes(b) for b, num in json.load(file).items()}


        with open(merges_filepath, 'r') as file:
            merges: Dict[Tuple[int, int], int] = {}
            for line in file:
                merge = line.rstrip().split(" ")
                merges[(int(merge[0]), int(merge[1]))] = int(merge[2])
        params = BPETokenizerParams(vocab, merges)
        return cls(params, special_tokens)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def encode(self, text: str) -> List[int]:
        split_text = [text]
        if self.special_tokens: # split by special tokens first
            special_split = r"(" + r"|".join(re.escape(tok) for tok in sorted(self.special_tokens, reverse=True)) + r")" #+  PAT
            split_text: List[str] = [string for string in re.split(special_split, text) if len(string)] # get rid of empty strings

        pretokenized_text: List[List[bytes]] = [] # list of list of bytes. inner lists are mostly just individual bytes except special tokens which are already fully formed
        # print(self.reverse_vocab)
        for t in tqdm(split_text, desc="Pretokenizing documents"):
        # for t in split_text:
            if self.special_tokens and t in self.special_tokens:
                pretokenized_text.append([self.reverse_vocab[t.encode("utf-8")]])
            else:
                list_of_bytes: List[bytes] = [string.encode("utf-8") for string in re.findall(PAT, t)]
                list_of_list_of_bytes: List[List[bytes]]= [[self.reverse_vocab[bytes([b])] for b in bs] for bs in list_of_bytes]
                pretokenized_text += list_of_list_of_bytes

        inds: List[int] = [] # token numbers

        for token in tqdm(pretokenized_text, desc="Merging tokens"):
        # for token in pretokenized_text:
            merges_to_perform = {} # index to order
            while True: # merging
                for i in range(len(token) - 1): # find all merges
                    curr_merge =(token[i], token[i+1])
                    if curr_merge in self.params.merges:
                        merges_to_perform[i] = self.params.merges[curr_merge]
                if merges_to_perform: # do first merge that appears in merges
                    best_merge_index = min(merges_to_perform, key=lambda x: merges_to_perform[x])
                    token[best_merge_index] = self.params.merges[token[best_merge_index], token[best_merge_index+1]]
                    token.pop(best_merge_index+1)
                    merges_to_perform.clear()
                else:
                    break
            inds += token
   
        return inds

    def decode(self, indices: List[int]) -> str:
        bytes_list: List[bytes] = [self.params.vocab[i] for i in indices] # list of every index converted to bytes
        text = b''.join(bytes_list).decode("utf-8", errors="replace") # join bytes into one string then decode
        return text



# worker function to process each chunk
def process_chunk(start, end, special_tokens, file_path):
    """Process a single chunk of text for BPE training.
    Remove special tokens, pre-tokenize the chunk, then count the frequency of each byte pair.
    
    Args:
        start: start position in the file
        end: end position in the file
        special_tokens: List of special tokens to handle
        file_path: Path to the input file
    """
    byte_pairs = Counter()
    freq_table = defaultdict(int)
    
    # Pre-compile patterns for better performance
    escaped_special_tokens = [re.escape(token) for token in special_tokens]
    split_pattern = "|".join(escaped_special_tokens)
    
    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start)
        chunk = chunk.decode("utf-8", errors="ignore")
        
        # Split on special tokens and process each part separately
        parts = [p for p in re.split(split_pattern, chunk) if p.strip()]
        
        # Process each part separately to prevent merging across special tokens
        for part in parts:
            # Pre-tokenize the part
            for word in PAT.finditer(part):
                word_text = word.group()
                
                # Process word bytes (count byte pairs and tuples) 
                word_bytes = word_text.encode('utf-8')
                word_tuple = tuple(bytes([b]) for b in word_bytes)
                
                if word_tuple:
                    freq_table[word_tuple] += 1
                    
                    # Process byte pairs
                    word_len = len(word_bytes)
                    for i in range(word_len - 1):
                        pair = (bytes([word_bytes[i]]), bytes([word_bytes[i + 1]]))
                        byte_pairs[pair] += 1
    
    return byte_pairs, freq_table

def process_chunk_wrapper(args):
    """Wrapper function to unpack arguments for process_chunk.
    
    Args:
        args: Tuple containing (start, end, special_tokens, file_path)
    """
    start, end, special_tokens, file_path = args
    return process_chunk(start, end, special_tokens, file_path)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.
    
    Args:
        input_path: Path to input text file
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens to handle
    """
    # Initialize vocab with special tokens
    vocab = {} # dict[int, bytes]
    merges = [] # list[tuple[bytes, bytes]]
    
    print("Initializing vocabulary with special tokens...")
    # Add special tokens to vocab first
    for i, token in enumerate(special_tokens):
        vocab[i] = token.encode('utf-8')
    
    print("Adding byte vocabulary...")
    # Add byte vocabulary after special tokens
    for i in range(256):
        vocab[len(special_tokens) + i] = bytes([i])
    
    # Set fixed number of chunks
    num_chunks = 16
    file_size = os.path.getsize(input_path)
    print(f"Attempting to split {file_size/1024/1024:.2f}MB file into {num_chunks} chunks")
    
    print("Reading input text and finding chunk boundaries...")
    # Read input text and get chunk boundaries that don't split special tokens
    total_byte_pairs = Counter()
    total_freq_table = defaultdict(int)


    print("Input path: ",input_path)
    with open(input_path, 'rb') as f:
        # find_chunk_boundaries ensures chunks start at special tokens
        # may return fewer chunks if boundaries would overlap
        boundaries = find_chunk_boundaries(
            f, num_chunks, "<|endoftext|>".encode("utf-8"))
        print(f"Found {len(boundaries)-1} valid chunk boundaries", boundaries)
        
        # Create list of (start, end) pairs to process
        chunk_pairs = list(zip(boundaries[:-1], boundaries[1:]))

        '''
        # Process each chunk sequentially
        for start, end in tqdm(zip(boundaries[:-1], boundaries[1:]), total=len(boundaries)-1):
            byte_pairs, freq_table = process_chunk(start, end, special_tokens, input_path)
            total_byte_pairs.update(byte_pairs)
            for word_tuple, count in freq_table.items():
                total_freq_table[word_tuple] += count
                
        '''

        
        print("Processing chunks with multiprocessing pool...")
        # Create a pool of worker processes with a reasonable number of processes
        # Use number of CPU cores as a reasonable default
        num_processes = min(multiprocessing.cpu_count(), len(chunk_pairs))
        print(f"Using {num_processes} processes")
        
        # Prepare arguments for each chunk
        chunk_args = [(start, end, special_tokens, input_path) for start, end in chunk_pairs]
        
        # Process chunks with progress bar
        with multiprocessing.Pool(processes=num_processes) as pool:
            for byte_pairs, freq_table in tqdm(
                pool.imap_unordered(process_chunk_wrapper, chunk_args),
                total=len(chunk_args),
                desc="Processing chunks"
            ):
                total_byte_pairs.update(byte_pairs)
                for word_tuple, count in freq_table.items():
                    total_freq_table[word_tuple] += count

    print("Beginning BPE merges...")
    print(f"Target vocabulary size: {vocab_size}")
    # perform bpe merge until vocab size is reached 
    current_vocab_size = len(vocab)
    
    # Pre-allocate lists for faster updates
    updates = []
    word_tuples = []
    
    while current_vocab_size < vocab_size:
        if not total_byte_pairs:  # No more pairs to merge
            print("No more pairs to merge, stopping early")
            break
            
        # Get pair with highest count, breaking ties by taking lexicographically greater pair
        # Optimize max operation by using a single pass
        best_pair = None
        best_count = -1
        for pair, count in total_byte_pairs.items():
            if count > best_count or (count == best_count and pair > best_pair):
                best_pair = pair
                best_count = count
        
        # Skip merging if either token contains a special token pattern
        if any(b"<|" in token for token in best_pair):
            del total_byte_pairs[best_pair]
            continue
            
        # add merged pair to vocab
        merged_bytes = bytes(best_pair[0] + best_pair[1])
        vocab[current_vocab_size] = merged_bytes
        merges.append(best_pair)  # Store the pair that was merged

        # update freq table and byte pairs
        updates.clear()
        word_tuples.clear()
        word_tuples.extend(total_freq_table.keys())
        
        for word_tuple in word_tuples:
            # Find all occurrences of the best pair in the word
            matching_positions = []
            word_len = len(word_tuple)
            for i in range(word_len - 1):
                if word_tuple[i] == best_pair[0] and word_tuple[i + 1] == best_pair[1]:
                    matching_positions.append(i)
            
            if matching_positions:
                # Create new word tuple with merged pairs
                new_word = list(word_tuple)
                for pos in reversed(matching_positions):
                    new_word[pos:pos+2] = [merged_bytes]
                new_word = tuple(new_word)
                updates.append((word_tuple, new_word, total_freq_table[word_tuple]))

        # Apply updates
        for old_tuple, new_tuple, freq in updates:
            # Update frequency table
            del total_freq_table[old_tuple]
            total_freq_table[new_tuple] = freq
            
            # Batch update byte pairs
            # Remove old pairs
            old_pairs = [(old_tuple[i], old_tuple[i + 1]) 
                        for i in range(len(old_tuple) - 1)]
            for pair in old_pairs:
                total_byte_pairs[pair] -= freq
                if total_byte_pairs[pair] <= 0:
                    del total_byte_pairs[pair]
            
            # Add new pairs
            new_pairs = [(new_tuple[i], new_tuple[i + 1]) 
                        for i in range(len(new_tuple) - 1)]
            for pair in new_pairs:
                total_byte_pairs[pair] += freq

        # Remove the merged pair from consideration
        if best_pair in total_byte_pairs:
            del total_byte_pairs[best_pair]
        
        current_vocab_size += 1
    
    print(f"Final vocabulary size: {current_vocab_size}")

    return vocab, merges

'''
if __name__ == "__main__":
    # import arg 
    import argparse
    import time
    import psutil

    parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
    parser.add_argument('--dataset', type=str, default = 'TS', choices = ['TS', 'OWT'])
    
    args = parser.parse_args()

    if args.dataset == 'TS':
        # TS 
        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)

        vocab, merges = run_train_bpe('data/TinyStoriesV2-GPT4-train.txt', 10000, ['<|endoftext|>'])

        end_time = time.time()
        final_memory = process.memory_info().rss / (1024 * 1024)

        time_taken = end_time - start_time
        memory = final_memory - initial_memory

        print("TS")
        print("time_taken: ", time_taken)
        print("max_memory (MB): ", memory)
        # print the longest token in the vocabulary
        print("Longest token: ", max(vocab.values(), key=len))
    
    elif args.dataset == 'OWT':
        # OWT 
        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)

        vocab, merges = run_train_bpe('data/owt_train.txt',  32000, ['<|endoftext|>'])

        end_time = time.time()
        final_memory = process.memory_info().rss / (1024 * 1024)

        time_taken = end_time - start_time
        memory = final_memory - initial_memory

        print("OWT")
        print("time_taken: ", time_taken)
        print("max_memory (MB): ", memory)
        # print the longest token in the vocabulary
        print("Longest token: ", max(vocab.values(), key=len))

'''
# vocab, merges = run_train_bpe('cs336_basics/temp.txt', (256+1+6), ['<|endoftext|>'])

# print(vocab)
# print(merges)

# vocab, merges = run_train_bpe('cs336_basics/TinyStoriesV2-GPT4-valid.txt', 1000, ['<|endoftext|>'])
# # # run_train_bpe('cs336_basics/TinyStoriesV2-GPT4-valid.txt', 1000, ['<|endoftext|>'])

# print(vocab)
# print(merges)
