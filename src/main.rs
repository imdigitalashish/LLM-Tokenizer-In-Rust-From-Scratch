mod bpe_tokenizer;
use bpe_tokenizer::RustyBPETokenizer;
use std::io::{self, Read};
use std::path::Path;

use std::{collections::HashSet, fs::File};


fn read_file<P: AsRef<Path>>(path: P) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}



fn main() {

    let training_text = read_file("./the-verdict.txt").unwrap();

    let mut tokenizer = RustyBPETokenizer::new();
    
    let mut allowed_special = HashSet::new();
    allowed_special.insert("<|endoftext|>".to_string());
    
    tokenizer.train(&training_text, 1000, allowed_special);
    
    tokenizer.save_vocab_and_merges("./vacob.txt", "./bpe_merges.txt").unwrap();
    
    let input_text = "Jack embraced beauty through art and life.";
    let token_ids = tokenizer.encode(input_text);
    println!("{:?}", token_ids);
    
    // Test decoding
    match tokenizer.decode(&token_ids) {
        Ok(decoded) => println!("Decoded: {}", decoded),
        Err(e) => println!("Error decoding: {}", e),
    }
}