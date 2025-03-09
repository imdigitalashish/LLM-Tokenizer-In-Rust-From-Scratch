# 🦀 Rusty BPE Tokenizer

A simple but powerful Byte Pair Encoding (BPE) tokenizer written in Rust.

## What is this?

This is a lightweight implementation of a BPE tokenizer that's perfect for text processing, natural language tasks, and machine learning applications. If you've ever needed to turn text into tokens that models can understand, this is your friend!

## Why BPE?

BPE (Byte Pair Encoding) is the secret sauce behind how many language models understand text. It works by:

1. Starting with individual characters
2. Finding the most common adjacent pairs 
3. Merging them into new tokens
4. Repeating until we have a nice vocabulary size

This approach handles rare words and new words gracefully, giving you the best of both character-level and word-level tokenization.

## Features

- ⚡ Fast tokenization from scratch
- 🔄 Compatible with OpenAI-style vocab files
- 🧠 Train your own custom tokenizer
- 💾 Save and load vocabularies
- 🛠️ Full encode/decode functionality
- 🧩 Special token support

## Quick Start

```rust
use bpe_tokenizer::BPETokenizerSimple;
use std::collections::HashSet;

fn main() {
    // Create a new tokenizer
    let mut tokenizer = BPETokenizerSimple::new();
    
    // Train from scratch
    let text = "Hello world! This is a sample text to train our tokenizer.";
    let vocab_size = 100;
    let special_tokens = HashSet::from(["<UNK>".to_string(), "<PAD>".to_string()]);
    
    tokenizer.train(text, vocab_size, special_tokens);
    
    // Encode text to token IDs
    let tokens = tokenizer.encode("Hello world!");
    println!("Tokens: {:?}", tokens);
    
    // Decode token IDs back to text
    let decoded = tokenizer.decode(&tokens).unwrap();
    println!("Decoded: {}", decoded);
    
    // Save for later
    tokenizer.save_vocab_and_merges("vocab.json", "merges.json").unwrap();
}
```

## Loading Pre-trained Tokenizers

Already have a vocabulary and merges file? No problem:

```rust
// Load OpenAI-style files
tokenizer.load_vocab_and_merges_from_openai("vocab.json", "merges.txt").unwrap();

// Or load our own format
tokenizer.load_vocab_and_merges("vocab.json", "merges.json").unwrap();
```

## How It Works

The tokenizer works in three main phases:

1. **Training**: Analyzes text to find common character pairs
2. **Encoding**: Converts text into token IDs using learned merges
3. **Decoding**: Converts token IDs back into readable text

During encoding, the tokenizer adds a special 'Ġ' character at the beginning of words to preserve whitespace information.

## Use Cases

- Text preprocessing for machine learning
- Building custom language models
- Efficient text compression
- Token-level text analysis


## Contributing

Found a bug? Have an idea? Feel free to open an issue or submit a PR!


---

Happy tokenizing! 🚀

Best Regards,
Ashish Kumar Verma ( You're Friendly Neighbourhood Tinkerer )