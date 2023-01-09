use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde_json::json;
use serde_json::Value;
use tokenizers::tokenizer::Tokenizer;
use tokenizers::{Encoding, PaddingDirection, PaddingParams, PaddingStrategy};

use crate::error::{Error, Result};

/// A simple tokenizer wrapper that reads special token map for further use in the pipeline.
pub struct AutoTokenizer {
    pub tokenizer: Tokenizer,
    pub eos_token: String,
    pub pad_token: String,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
}

impl AutoTokenizer {
    pub fn new_from_memory(
        tokenizer_config: String,
        special_tokens_map: String,
    ) -> Result<AutoTokenizer> {
        let mut tok = Tokenizer::from_str(tokenizer_config.as_ref())?;
        let special_tokens_map: Value = serde_json::from_str(special_tokens_map.as_ref())?;
        let special_tokens_map = special_tokens_map.as_object().ok_or(Error::GenericError {
            message: "Special tokens map should be a JSON object".to_string(),
        })?;
        let eos_token: String;
        let eos_token_id: u32;
        if special_tokens_map.contains_key("eos_token") {
            eos_token = special_tokens_map["eos_token"]
                .as_str()
                .unwrap()
                .to_string();
            eos_token_id = tok.token_to_id(&eos_token).unwrap()
        } else if special_tokens_map.contains_key("sep_token") {
            eos_token = special_tokens_map["sep_token"]
                .as_str()
                .unwrap()
                .to_string();
            eos_token_id = tok.token_to_id(&eos_token).unwrap()
        } else if special_tokens_map.contains_key("eos_token_id") {
            eos_token_id = special_tokens_map["eos_token_id"].as_i64().unwrap() as u32;
            eos_token = tok.id_to_token(eos_token_id).unwrap()
        } else if special_tokens_map.contains_key("sep_token_id") {
            eos_token_id = special_tokens_map["sep_token_id"].as_i64().unwrap() as u32;
            eos_token = tok.id_to_token(eos_token_id).unwrap()
        } else {
            return Err(Error::GenericError {
                message: "No eos token found in special tokens map".to_string(),
            });
        }
        let pad_token = match special_tokens_map.get("pad_token") {
            Some(v) => v.as_str().unwrap().to_string(),
            None => match special_tokens_map.get("pad_token_id") {
                Some(v) => tok.id_to_token(v.as_i64().unwrap() as u32).unwrap(),
                None => "[PAD]".to_string(),
            },
        };
        let pad_token_id = tok.token_to_id(&pad_token).unwrap();

        if let None = tok.get_padding() {
            tok.with_padding(Option::from(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                direction: PaddingDirection::Right,
                pad_to_multiple_of: None,
                pad_id: pad_token_id,
                pad_type_id: 0,
                pad_token: pad_token.clone(),
            }));
        }
        Ok(AutoTokenizer {
            tokenizer: tok,
            eos_token,
            pad_token,
            eos_token_id,
            pad_token_id,
        })
    }

    pub fn new(
        tokenizer_config_path: PathBuf,
        special_tokens_map_path: PathBuf,
    ) -> Result<AutoTokenizer> {
        let tokenizer_config = std::fs::read_to_string(tokenizer_config_path)?;
        let special_tokens_map = std::fs::read_to_string(special_tokens_map_path)?;
        AutoTokenizer::new_from_memory(tokenizer_config, special_tokens_map)
    }
}
