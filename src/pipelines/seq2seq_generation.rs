use std::path::{Path, PathBuf};
use std::sync::Arc;

use ort::environment::Environment;
use ndarray::{concatenate, s, Array, Array1, Array2, ArrayView1, Axis, Ix2};
use ort::GraphOptimizationLevel;
use tokenizers::Encoding;

use crate::common::Device;
use crate::error::Result;
use crate::hf_hub::hf_hub_download;
use crate::sampling::Sampler;
use crate::tokenizer::AutoTokenizer;
use crate::Seq2SeqGenerationModel;

/// Wraps Huggingface AutoModelForCausalLM exported to ONNX with `seq2seq-lm` feature
/// and pretrained tokenizer into simple text to text generative interface.
///
/// !!! Note
///     Uses bos_token if it exists to start decoder generation and eos_token to stop generation.
///     If bos_token is not in special_tokens_map, then eos_token is used.
///     If you want to use a different first decoder token, provide decoder_prompt parameter.
///     No additional special tokens are added to decoder_prompt.
///
/// Export docs https://huggingface.co/docs/transformers/serialization#export-to-onnx
///
/// # Example
///
///```no_run
/// use std::fs;
/// use ort::{GraphOptimizationLevel, LoggingLevel};
/// use ort::environment::Environment;
/// use edge_transformers::{Seq2SeqGenerationPipeline, TopKSampler,Device};
///
/// let environment = Environment::builder()
///   .with_name("test")
///  .with_log_level(LoggingLevel::Verbose)
/// .build()
/// .unwrap();
///
/// let sampler = TopKSampler::new(50, 0.9);
/// let pipeline = Seq2SeqGenerationPipeline::from_pretrained(
///     environment.into_arc(),
///    "optimum/t5-small".to_string(),
///     Device::CPU,
///     GraphOptimizationLevel::Level3,
/// ).unwrap();
///
/// let input = "This is a test";
///
/// println!("{}", pipeline.generate(input, None, 10, &sampler).unwrap());
/// ```
pub struct Seq2SeqGenerationPipeline<'a> {
    tokenizer: AutoTokenizer,
    model: Seq2SeqGenerationModel<'a>,
    token_type_support: bool,
    decoder_type_support: bool,
}

impl<'a> Seq2SeqGenerationPipeline<'a> {
    pub fn from_pretrained(
        env: Arc<Environment>,
        model_id: String,
        device: Device,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<Self> {
        let model_dir = Path::new(&model_id);
        if model_dir.exists() {
            let model_path = model_dir.join("model.onnx");
            let tokenizer_path = model_dir.join("tokenizer.json");
            let mut special_tokens_path = model_dir.join("special_tokens_map.json");
            if !special_tokens_path.exists() {
                special_tokens_path = model_dir.join("config.json");
            }
            Self::new_from_files(
                env,
                model_path,
                tokenizer_path,
                special_tokens_path,
                device,
                optimization_level,
            )
        } else {
            let model_path = hf_hub_download(&model_id, "model.onnx", None, None)?;
            let tokenizer_path = hf_hub_download(&model_id, "tokenizer.json", None, None)?;
            let mut special_tokens_path =
                hf_hub_download(&model_id, "special_tokens_map.json", None, None);
            if special_tokens_path.is_err() {
                special_tokens_path = hf_hub_download(&model_id, "config.json", None, None);
            }
            Self::new_from_files(
                env,
                model_path,
                tokenizer_path,
                special_tokens_path?,
                device,
                optimization_level,
            )
        }
    }

    /// Creates new pipeline from ONNX model bytes and tokenizer configuration.
    pub fn new_from_memory(
        environment: Arc<Environment>,
        model: &'a [u8],
        tokenizer_config: String,
        special_tokens_map: String,
        device: Device,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<Self> {
        let tokenizer = AutoTokenizer::new_from_memory(tokenizer_config, special_tokens_map)?;
        let model = Seq2SeqGenerationModel::new_from_memory(
            environment,
            model,
            device,
            optimization_level,
        )?;
        let token_type_support = model.get_token_type_support();
        let decoder_type_support = model.get_decoder_token_type_support();

        Ok(Seq2SeqGenerationPipeline {
            tokenizer,
            model,
            token_type_support,
            decoder_type_support,
        })
    }

    /// Creates new pipeline from ONNX model file and tokenizer configuration.
    pub fn new_from_files(
        environment: Arc<Environment>,
        model: PathBuf,
        tokenizer_config: PathBuf,
        special_tokens_map: PathBuf,
        device: Device,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<Self> {
        let tokenizer = AutoTokenizer::new(tokenizer_config, special_tokens_map)?;
        let model =
            Seq2SeqGenerationModel::new_from_file(environment, model, device, optimization_level)?;
        let token_type_support = model.get_token_type_support();
        let decoder_type_support = model.get_decoder_token_type_support();
        Ok(Seq2SeqGenerationPipeline {
            tokenizer,
            model,
            token_type_support,
            decoder_type_support,
        })
    }

    pub fn generate<'sampler>(
        &self,
        prompt: &str,
        decoder_prompt: Option<&str>,
        max_length: i32,
        sampler: &'sampler dyn Sampler,
    ) -> Result<String> {
        let prompt_batch = vec![prompt.to_string()];
        let decoder_prompt_batch = match decoder_prompt {
            Some(decoder_prompt) => Some(vec![decoder_prompt.to_string()]),
            None => None,
        };
        self.generate_batch(prompt_batch, decoder_prompt_batch, max_length, sampler)
            .map(|batch| batch[0].clone())
    }

    /// Generates text from input batch text.
    pub fn generate_batch<'sampler>(
        &self,
        prompt: Vec<String>,
        decoder_prompt: Option<Vec<String>>,
        max_length: i32,
        sampler: &'sampler dyn Sampler,
    ) -> Result<Vec<String>> {
        let batch_size = prompt.len();
        let encoding = self.tokenizer.tokenizer.encode_batch(prompt, true)?;
        let decoder_encoding = match decoder_prompt {
            Some(decoder_prompt) => {
                let decoder_encoding = self
                    .tokenizer
                    .tokenizer
                    .encode_batch(decoder_prompt, true)?;
                decoder_encoding
            }
            None => {
                let decoder_encoding = self.tokenizer.tokenizer.encode_batch(
                    vec![self.tokenizer.eos_token.to_string(); batch_size],
                    false,
                )?;
                decoder_encoding
            }
        };
        if batch_size != decoder_encoding.len() {
            return Err("Prompt and decoder prompt batch size must be equal"
                .to_string()
                .into());
        }
        let mut generated_ids = vec![vec![]; encoding.len()];
        let mut eos_token_generated = vec![false; encoding.len()];

        let enc_tuple = self.enc_vec_to_tensor(encoding);
        let input_ids = enc_tuple.0;
        let attention_mask = enc_tuple.1;
        let mut type_ids = enc_tuple.2;

        let dec_tuple = self.enc_vec_to_tensor(decoder_encoding);
        let mut decoder_input_ids = dec_tuple.0;
        let mut decoder_attention_mask = dec_tuple.1;
        let mut decoder_type_ids = dec_tuple.2;

        for _ in 0..max_length {
            let output = self.model.forward(
                input_ids.to_owned(),
                Some(attention_mask.to_owned()),
                decoder_input_ids.to_owned(),
                Some(decoder_attention_mask.to_owned()),
                Some(type_ids.to_owned()),
                Some(decoder_type_ids.to_owned()),
            )?;
            let seq_len = output.shape()[1];
            let logits = output.index_axis_move(Axis(1), seq_len - 1);
            let next_tokens = Array::from_iter(sampler.sample(logits.view()));
            decoder_input_ids = concatenate(
                Axis(1),
                &[
                    decoder_input_ids.view(),
                    next_tokens.view().insert_axis(Axis(1)),
                ],
            )?;
            decoder_attention_mask = concatenate(
                Axis(1),
                &[
                    decoder_attention_mask.view(),
                    Array2::ones((batch_size, 1)).view(),
                ],
            )?;
            if self.token_type_support {
                type_ids = concatenate(
                    Axis(1),
                    &[
                        type_ids.view(),
                        (Array2::ones((batch_size, 1))
                            * type_ids.slice(s![.., type_ids.shape()[1]]))
                        .view(),
                    ],
                )?;
            }
            if self.decoder_type_support {
                decoder_type_ids = concatenate(
                    Axis(1),
                    &[
                        decoder_type_ids.view(),
                        (Array2::ones((batch_size, 1))
                            * decoder_type_ids.slice(s![.., decoder_type_ids.shape()[1]]))
                        .view(),
                    ],
                )?;
            }
            generated_ids = generated_ids
                .iter()
                .zip(next_tokens.iter())
                .zip(eos_token_generated.iter())
                .map(|((ids, token), eos_generated)| {
                    if *eos_generated {
                        ids.clone()
                    } else {
                        let mut ids = ids.clone();
                        ids.push(*token);
                        ids
                    }
                })
                .collect();

            eos_token_generated = next_tokens
                .iter()
                .zip(eos_token_generated.iter())
                .map(|(t, e)| e | (*t == self.tokenizer.eos_token_id))
                .collect();
            if eos_token_generated.iter().all(|e| *e) {
                break;
            }
        }
        let sentences = self.tokenizer.tokenizer.decode_batch(generated_ids, true)?;
        Ok(sentences)
    }

    fn enc_vec_to_tensor(
        &self,
        encodings: Vec<Encoding>,
    ) -> (Array<u32, Ix2>, Array<u32, Ix2>, Array<u32, Ix2>) {
        let input_ids_tensors = &encodings
            .iter()
            .map(|e| Array1::from_iter(e.get_ids().iter().map(|i| *i as u32)))
            .collect::<Vec<Array1<u32>>>();

        let att_mask_tensors = &encodings
            .iter()
            .map(|e| Array1::from_iter(e.get_attention_mask().iter().map(|i| *i as u32)))
            .collect::<Vec<Array1<u32>>>();

        let type_ids_tensors = &encodings
            .iter()
            .map(|e| Array1::from_iter(e.get_type_ids().iter().map(|i| *i as u32)))
            .collect::<Vec<Array1<u32>>>();

        let input_ids = ndarray::stack(
            Axis(0),
            input_ids_tensors
                .iter()
                .map(|e| e.view())
                .collect::<Vec<ArrayView1<_>>>()
                .as_slice(),
        )
        .unwrap();

        let attention_mask = ndarray::stack(
            Axis(0),
            att_mask_tensors
                .iter()
                .map(|e| e.view())
                .collect::<Vec<ArrayView1<_>>>()
                .as_slice(),
        )
        .unwrap();

        let type_ids = ndarray::stack(
            Axis(0),
            type_ids_tensors
                .iter()
                .map(|e| e.view())
                .collect::<Vec<ArrayView1<_>>>()
                .as_slice(),
        )
        .unwrap();

        (input_ids, attention_mask, type_ids)
    }
}

#[cfg(test)]
mod tests {
    use ort::environment::Environment;
    use ort::{GraphOptimizationLevel, LoggingLevel};

    use crate::common::Device;
    use crate::error::Result;
    use crate::sampling::TopKSampler;
    use crate::Seq2SeqGenerationPipeline;

    #[ignore]
    #[test]
    fn test_gen_model() -> Result<()> {
        let env = Environment::builder()
            .with_log_level(LoggingLevel::Error)
            .build()
            .unwrap();
        let pipeline = Seq2SeqGenerationPipeline::from_pretrained(
            env.into_arc(),
            "optimum/t5-small".to_string(),
            Device::CPU,
            GraphOptimizationLevel::Level3,
        )?;
        let sampler = TopKSampler::new(5, 1.0);
        let output = pipeline.generate("Hello world mate", None, 10, &sampler)?;
        println!("{:?}", output);
        Ok(())
    }
}
