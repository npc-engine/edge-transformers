use std::path::{Path, PathBuf};

use onnxruntime::{GraphOptimizationLevel, ndarray};
use onnxruntime::environment::Environment;
use onnxruntime::ndarray::{
    Array, Array1, ArrayView1, Axis, Ix2,
};
use tokenizers::Encoding;

use crate::{clone, Seq2SeqDecoderModelWithPKVs};
use crate::common::Device;
use crate::error::Result;
use crate::hf_hub::hf_hub_download;
use crate::sampling::Sampler;
use crate::seq2seq_encoder::Seq2SeqEncoderModel;
use crate::tokenizer::AutoTokenizer;

/// Wraps Huggingface Optimum pipeline export to ONNX with `seq2seq-lm-with-past` task.
///
/// Takes more memory than `OptimumSeq2SeqGenerationModel` (exactly by 1 decoder model size)
/// but generation is faster.
///
/// !!! Note
///     Uses eos_token as first decoder token.
///     If you want to use a different first decoder token, provide decoder_prompt parameter.
///     No additional special tokens are added to decoder_prompt.
///
/// # Export
///
/// Example command:
///
/// ```bash
/// pip install optimum[onnxruntime]
/// python -m optimum.exporters.onnx --model t5-base --for-ort --task seq2seq-lm-with-past ./resources/t5-base-nopast
/// ```
///
/// See details in [Optimum docs](https://huggingface.co/docs/optimum/onnxruntime/overview).
///
/// # Example
///
/// ```
/// use std::fs;
/// use onnxruntime::{GraphOptimizationLevel, LoggingLevel};
/// use onnxruntime::environment::Environment;
/// use edge_transformers::{OptimumSeq2SeqPipelineWithPKVs, TopKSampler,Device};
///
/// let environment = Environment::builder()
///   .with_name("test")
///  .with_log_level(LoggingLevel::Verbose)
/// .build()
/// .unwrap();
///
/// let sampler = TopKSampler::new(50, 0.9);
/// let pipeline = OptimumSeq2SeqPipelineWithPKVs::from_pretrained(
///     &environment,
///     "optimum/t5-small".to_string(),
///     Device::CPU,
///     GraphOptimizationLevel::All,
/// ).unwrap();
///
/// let input = "This is a test";
///
/// println!("{}", pipeline.generate(input, None, 10, &sampler).unwrap());
/// ```
pub struct OptimumSeq2SeqPipelineWithPKVs<'a> {
    tokenizer: AutoTokenizer,
    encoder_model: Seq2SeqEncoderModel<'a>,
    decoder_model: Seq2SeqDecoderModelWithPKVs<'a>,
}

impl<'a> OptimumSeq2SeqPipelineWithPKVs<'a> {
    pub fn from_pretrained(
        env: &'a Environment,
        model_id: String,
        device: Device,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<Self> {
        let model_dir = Path::new(&model_id);
        if model_dir.exists() {
            let decoder_model_path = model_dir.join("decoder_model.onnx");
            let decoder_with_past_model = model_dir.join("decoder_with_past_model.onnx");
            let encoder_model_path = model_dir.join("encoder_model.onnx");
            let tokenizer_path = model_dir.join("tokenizer.json");
            let mut special_tokens_path = model_dir.join("special_tokens_map.json");
            if !special_tokens_path.exists() {
                println!("special_tokens_map.json not found, trying to use config.json");
                special_tokens_path = model_dir.join("config.json");
            } else {
                println!("special_tokens_map.json found");
            }
            Self::new_from_files(
                env,
                encoder_model_path,
                decoder_model_path,
                decoder_with_past_model,
                tokenizer_path,
                special_tokens_path,
                device,
                optimization_level,
            )
        } else {
            let decoder_model_path = hf_hub_download(&model_id, "decoder_model.onnx", None, None)?;
            let encoder_model_path = hf_hub_download(&model_id, "encoder_model.onnx", None, None)?;
            let decoder_with_past_model =
                hf_hub_download(&model_id, "decoder_with_past_model.onnx", None, None)?;
            let tokenizer_path = hf_hub_download(&model_id, "tokenizer.json", None, None)?;
            let mut special_tokens_path =
                hf_hub_download(&model_id, "special_tokens_map.json", None, None);
            if special_tokens_path.is_err() {
                special_tokens_path = hf_hub_download(&model_id, "config.json", None, None);
            }
            Self::new_from_files(
                env,
                encoder_model_path,
                decoder_model_path,
                decoder_with_past_model,
                tokenizer_path,
                special_tokens_path?,
                device,
                optimization_level,
            )
        }
    }

    /// Creates new pipeline from ONNX model bytes and tokenizer configuration.
    pub fn new_from_memory(
        environment: &'a Environment,
        encoder_model: &[u8],
        decoder_model: &[u8],
        decoder_model_pkvs: &[u8],
        tokenizer_config: String,
        special_tokens_map: String,
        device: Device,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<Self> {
        let tokenizer = AutoTokenizer::new_from_memory(tokenizer_config, special_tokens_map)?;
        let decoder_model = Seq2SeqDecoderModelWithPKVs::new_from_memory(
            environment,
            decoder_model,
            decoder_model_pkvs,
            device.clone(),
            clone(&optimization_level),
        )?;
        let encoder_model = Seq2SeqEncoderModel::new_from_memory(
            environment,
            encoder_model,
            device,
            optimization_level,
        )?;

        Ok(OptimumSeq2SeqPipelineWithPKVs {
            tokenizer,
            decoder_model,
            encoder_model,
        })
    }

    /// Creates new pipeline from ONNX model file and tokenizer configuration.
    pub fn new_from_files(
        environment: &'a Environment,
        encoder_model: PathBuf,
        decoder_model: PathBuf,
        decoder_model_pkvs: PathBuf,
        tokenizer_config: PathBuf,
        special_tokens_map: PathBuf,
        device: Device,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<Self> {
        let tokenizer = AutoTokenizer::new(tokenizer_config, special_tokens_map)?;
        let encoder_model = Seq2SeqEncoderModel::new_from_file(
            environment,
            encoder_model,
            device.clone(),
            clone(&optimization_level),
        )?;
        let decoder_model = Seq2SeqDecoderModelWithPKVs::new_from_file(
            environment,
            decoder_model,
            decoder_model_pkvs,
            device,
            optimization_level,
        )?;
        Ok(OptimumSeq2SeqPipelineWithPKVs {
            tokenizer,
            encoder_model,
            decoder_model,
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
        let type_ids = enc_tuple.2;

        let encoder_hidden_state = self.encoder_model.forward(
            input_ids,
            Some(attention_mask.to_owned()),
            Some(type_ids),
        )?;

        let dec_tuple = self.enc_vec_to_tensor(decoder_encoding);
        let mut decoder_input_ids = dec_tuple.0;

        let mut past_key_values = None;

        for _ in 0..max_length {
            let (output, pkvs) = self.decoder_model.forward(
                decoder_input_ids.to_owned(),
                encoder_hidden_state.to_owned(),
                Some(attention_mask.to_owned()),
                past_key_values,
            )?;
            let seq_len = output.shape()[1];
            past_key_values = Some(pkvs);
            let logits = output.index_axis_move(Axis(1), seq_len - 1);
            let next_tokens = Array::from_iter(sampler.sample(logits.view()));
            decoder_input_ids = next_tokens.clone().insert_axis(Axis(1));

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
    use onnxruntime::{GraphOptimizationLevel, LoggingLevel};
    use onnxruntime::environment::Environment;

    use crate::OptimumSeq2SeqPipelineWithPKVs;
    use crate::common::Device;
    use crate::error::Result;
    use crate::sampling::TopKSampler;

    #[test]
    fn test_gen_model() -> Result<()> {
        let env = Environment::builder()
            .with_log_level(LoggingLevel::Error)
            .build()
            .unwrap();
        let pipeline = OptimumSeq2SeqPipelineWithPKVs::from_pretrained(
            &env,
            "optimum/t5-small".to_string(),
            Device::CPU,
            GraphOptimizationLevel::All,
        )?;
        let sampler = TopKSampler::new(5, 1.0);
        let output = pipeline.generate("Hello world mate", None, 10, &sampler)?;
        println!("{:?}", output);
        Ok(())
    }
}
