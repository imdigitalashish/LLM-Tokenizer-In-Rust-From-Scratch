use serde::{Deserialize, Serialize};
use serde_json;
use std::cmp::{max, min};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::iter::zip;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug)]
struct BpeMerge {
    pair: Vec<usize>,
    new_id: usize,
}

pub struct BPETokenizerSimple {
    vocab: HashMap<usize, String>,
    inverse_vocab: HashMap<String, usize>,
    bpe_merges: HashMap<(usize, usize), usize>,
}

impl BPETokenizerSimple {
    pub fn new() -> Self {
        BPETokenizerSimple {
            vocab: HashMap::new(),
            inverse_vocab: HashMap::new(),
            bpe_merges: HashMap::new(),
        }
    }

    // Inside the BPETokenizerSimple implementation, modify the train method:

    pub fn train(&mut self, text: &str, vocab_size: usize, allowed_special: HashSet<String>) {
        let mut processed_text = String::new();
        for (i, c) in text.chars().enumerate() {
            if c == ' ' && i != 0 {
                processed_text.push('Ġ');
            } else if c != ' ' {
                processed_text.push(c);
            }
        }

        let mut unique_chars: Vec<char> = (0..256)
            .map(|i| char::from_u32(i as u32).unwrap_or('�'))
            .collect();

        let mut additional_chars: Vec<char> = processed_text.chars().collect();
        additional_chars.sort();
        additional_chars.dedup();
        for c in additional_chars {
            if !unique_chars.contains(&c) {
                unique_chars.push(c);
            }
        }

        if !unique_chars.contains(&'Ġ') {
            unique_chars.push('Ġ');
        }

        for (i, c) in unique_chars.iter().enumerate() {
            self.vocab.insert(i, c.to_string());
            self.inverse_vocab.insert(c.to_string(), i);
        }

        for token in allowed_special {
            if !self.inverse_vocab.contains_key(&token) {
                let new_id = self.vocab.len();
                self.vocab.insert(new_id, token.clone());
                self.inverse_vocab.insert(token, new_id);
            }
        }

        let mut token_ids: Vec<usize> = Vec::new();
        for c in processed_text.chars() {
            match self.inverse_vocab.get(&c.to_string()) {
                Some(&id) => token_ids.push(id),
                None => {
                    println!(
                        "Warning: Character '{}' not found in vocabulary during training",
                        c
                    );
                    let new_id = self.vocab.len();
                    self.vocab.insert(new_id, c.to_string());
                    self.inverse_vocab.insert(c.to_string(), new_id);
                    token_ids.push(new_id);
                }
            }
        }

        // Modified section - Process merges more carefully
        for new_id in self.vocab.len()..vocab_size {
            let pair_id = Self::find_freq_pair(&token_ids, "most");
            if pair_id.is_none() {
                break;
            }

            let pair_id = pair_id.unwrap();

            // Before replacing the pair, add the merged token to vocabulary
            if let (Some(token0), Some(token1)) =
                (self.vocab.get(&pair_id.0), self.vocab.get(&pair_id.1))
            {
                let merged_token = format!("{}{}", token0, token1);
                self.vocab.insert(new_id, merged_token.clone());
                self.inverse_vocab.insert(merged_token, new_id);
                self.bpe_merges.insert(pair_id, new_id);

                // Only replace the pair in the token_ids after adding to vocab
                token_ids = Self::replace_pair(&token_ids, &pair_id, new_id);
            } else {
                println!(
                "Warning: Cannot merge - Token IDs ({}, {}) - one or both tokens not found in vocabulary",
                pair_id.0, pair_id.1
            );
                // Skip this merge, try the next most frequent pair
                continue;
            }
        }

        // No need for the second loop since we're adding to vocab during the merging process
    }

    pub fn load_vocab_and_merges_from_openai<P: AsRef<Path>>(
        &mut self,
        vocab_path: P,
        bpe_merges_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(vocab_path)?;
        let reader = BufReader::new(file);
        let loaded_vocab: HashMap<String, String> = serde_json::from_reader(reader)?;

        for (k, v) in loaded_vocab {
            let token_id = v.parse::<usize>()?;
            self.vocab.insert(token_id, k.clone());
            self.inverse_vocab.insert(k, token_id);
        }

        let file = File::open(bpe_merges_path)?;
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;

        let start_idx = if !lines.is_empty() && lines[0].starts_with("#") {
            1
        } else {
            0
        };

        for (rank, line) in lines[start_idx..].iter().enumerate() {
            let pair: Vec<&str> = line.trim().split_whitespace().collect();
            if pair.len() != 2 {
                println!("Line {} has more than 2 entries: {}", rank + 1, line.trim());
                continue;
            }

            let token1 = pair[0];
            let token2 = pair[1];

            if self.inverse_vocab.contains_key(token1) && self.inverse_vocab.contains_key(token2) {
                let token_id1 = *self.inverse_vocab.get(token1).unwrap();
                let token_id2 = *self.inverse_vocab.get(token2).unwrap();
                let merged_token = format!("{}{}", token1, token2);

                if self.inverse_vocab.contains_key(&merged_token) {
                    let merged_token_id = *self.inverse_vocab.get(&merged_token).unwrap();
                    self.bpe_merges
                        .insert((token_id1, token_id2), merged_token_id);
                } else {
                    println!(
                        "Merged token '{}' not found in vocab. Skipping.",
                        merged_token
                    );
                }
            } else {
                println!(
                    "Skipping pair ({}, {}) as one of the tokens is not in the vocabulary.",
                    token1, token2
                );
            }
        }

        Ok(())
    }
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();

        let processed_text = text.replace("\n", " \n ");
        let words: Vec<&str> = processed_text.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            if i > 0 && !word.starts_with('\n') {
                tokens.push(format!("Ġ{}", word));
            } else {
                tokens.push(word.to_string());
            }
        }

        let mut token_ids = Vec::new();
        for token in tokens {
            if let Some(&token_id) = self.inverse_vocab.get(&token) {
                token_ids.push(token_id);
            } else {
                match self.tokenize_with_bpe(&token) {
                    Ok(sub_token_ids) => token_ids.extend(sub_token_ids),
                    Err(e) => {
                        println!("Warning: Error tokenizing '{}': {}", token, e);

                        if let Some(&unk_id) = self.inverse_vocab.get("<UNK>") {
                            token_ids.push(unk_id);
                        } else {
                            for c in token.chars() {
                                if let Some(&char_id) = self.inverse_vocab.get(&c.to_string()) {
                                    token_ids.push(char_id);
                                }
                            }
                        }
                    }
                }
            }
        }

        token_ids
    }

    fn tokenize_with_bpe(&self, token: &str) -> Result<Vec<usize>, String> {
        let mut token_ids: Vec<usize> = Vec::new();
        let mut missing_chars = Vec::new();

        for c in token.chars() {
            if let Some(&id) = self.inverse_vocab.get(&c.to_string()) {
                token_ids.push(id);
            } else {
                missing_chars.push(c);
            }
        }

        if !missing_chars.is_empty() {
            return Err(format!(
                "Characters not found in vocab: {:?}",
                missing_chars
            ));
        }

        let mut can_merge = true;
        while can_merge && token_ids.len() > 1 {
            can_merge = false;
            let mut new_tokens = Vec::new();
            let mut i = 0;
            while i < token_ids.len() - 1 {
                let pair = (token_ids[i], token_ids[i + 1]);
                if let Some(&merged_token_id) = self.bpe_merges.get(&pair) {
                    new_tokens.push(merged_token_id);
                    i += 2;
                    can_merge = true;
                } else {
                    new_tokens.push(token_ids[i]);
                    i += 1;
                }
            }
            if i < token_ids.len() {
                new_tokens.push(token_ids[i]);
            }
            token_ids = new_tokens;
        }

        Ok(token_ids)
    }

    pub fn decode(&self, token_ids: &[usize]) -> Result<String, String> {
        let mut decoded_string = String::new();
        for &token_id in token_ids {
            if let Some(token) = self.vocab.get(&token_id) {
                if token.starts_with('Ġ') {
                    decoded_string.push(' ');
                    decoded_string.push_str(&token[token.chars().next().unwrap().len_utf8()..]);
                } else {
                    decoded_string.push_str(token);
                }
            } else {
                return Err(format!("Token ID {} not found in vocab.", token_id));
            }
        }
        Ok(decoded_string)
    }
    pub fn save_vocab_and_merges<P: AsRef<Path>>(
        &self,
        vocab_path: P,
        bpe_merges_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let vocab_file = File::create(vocab_path)?;
        serde_json::to_writer_pretty(vocab_file, &self.vocab)?;

        let merges_list: Vec<BpeMerge> = self
            .bpe_merges
            .iter()
            .map(|(&(p0, p1), &new_id)| BpeMerge {
                pair: vec![p0, p1],
                new_id,
            })
            .collect();

        let merges_file = File::create(bpe_merges_path)?;
        serde_json::to_writer_pretty(merges_file, &merges_list)?;

        Ok(())
    }

    pub fn load_vocab_and_merges<P: AsRef<Path>>(
        &mut self,
        vocab_path: P,
        bpe_merges_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Load vocabulary
        let vocab_file = File::open(vocab_path)?;
        let loaded_vocab: HashMap<String, String> = serde_json::from_reader(vocab_file)?;

        self.vocab.clear();
        self.inverse_vocab.clear();

        for (k, v) in loaded_vocab {
            let token_id = k.parse::<usize>()?;
            self.vocab.insert(token_id, v.clone());
            self.inverse_vocab.insert(v, token_id);
        }

        // Load BPE merges
        let merges_file = File::open(bpe_merges_path)?;
        let merges_list: Vec<BpeMerge> = serde_json::from_reader(merges_file)?;

        self.bpe_merges.clear();
        for merge in merges_list {
            if merge.pair.len() == 2 {
                self.bpe_merges
                    .insert((merge.pair[0], merge.pair[1]), merge.new_id);
            }
        }

        Ok(())
    }

    pub fn get_special_token_id(&self, token: &str) -> Option<usize> {
        self.inverse_vocab.get(token).copied()
    }

    fn find_freq_pair(token_ids: &[usize], mode: &str) -> Option<(usize, usize)> {
        if token_ids.len() < 2 {
            return None;
        }

        let mut pair_counts: HashMap<(usize, usize), usize> = HashMap::new();
        for pair in zip(token_ids.iter(), token_ids[1..].iter()) {
            let key = (*pair.0, *pair.1);
            *pair_counts.entry(key).or_insert(0) += 1;
        }

        if pair_counts.is_empty() {
            return None;
        }

        match mode {
            "most" => pair_counts
                .iter()
                .max_by_key(|&(_, count)| count)
                .map(|(&pair, _)| pair),
            "least" => pair_counts
                .iter()
                .min_by_key(|&(_, count)| count)
                .map(|(&pair, _)| pair),
            _ => panic!("Invalid mode. Choose 'most' or 'least'."),
        }
    }

    fn replace_pair(token_ids: &[usize], pair_id: &(usize, usize), new_id: usize) -> Vec<usize> {
        let mut dq: VecDeque<usize> = token_ids.iter().copied().collect();
        let mut replaced = Vec::new();

        while !dq.is_empty() {
            let current = dq.pop_front().unwrap();
            if !dq.is_empty() && (current, dq[0]) == *pair_id {
                replaced.push(new_id);
                dq.pop_front();
            } else {
                replaced.push(current);
            }
        }

        replaced
    }
}
