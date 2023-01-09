use std::path::{Path, PathBuf};

use onnxruntime::environment::Environment;
use onnxruntime::ndarray::{concatenate, s, Array, Array1, Array2, ArrayView1, Axis};
use onnxruntime::{ndarray, GraphOptimizationLevel};

use crate::common::Device;
use crate::error::Result;
use crate::hf_hub::hf_hub_download;
use crate::modeling::conditional_generation::ConditionalGenerationModel;
use crate::sampling::Sampler;
use crate::tokenizer::AutoTokenizer;

/// Wraps Huggingface Optimum pipeline exported to ONNX with `causal-lm` task.
///
/// !!! Note
///    Does not add any special tokens to the input text. If you want to add special tokens
///    to the input text, just provide them in the prompt.
///
/// Export docs https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model
///
/// # Example
///
/// ```
/// use std::fs;
/// use onnxruntime::{GraphOptimizationLevel, LoggingLevel};
/// use onnxruntime::environment::Environment;
/// use edge_transformers::{ConditionalGenerationPipeline, TopKSampler, Device};
///
/// let environment = Environment::builder()
///   .with_name("test")
///  .with_log_level(LoggingLevel::Verbose)
/// .build()
/// .unwrap();
///
/// let sampler = TopKSampler::new(50, 0.9);
/// let pipeline = ConditionalGenerationPipeline::from_pretrained(
///  &environment,
/// "optimum/gpt2".to_string(),
/// Device::CPU,
/// GraphOptimizationLevel::All,
/// ).unwrap();
///
/// let input = "This is a test";
///
/// println!("{}", pipeline.generate(input, 10, &sampler).unwrap());
/// ```
pub struct ConditionalGenerationPipeline<'a> {
    tokenizer: AutoTokenizer,
    model: ConditionalGenerationModel<'a>,
    token_type_support: bool,
}

impl<'a> ConditionalGenerationPipeline<'a> {
    pub fn from_pretrained(
        env: &'a Environment,
        model_id: String,
        device: Device,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<Self> {
        let model_dir = Path::new(&model_id);
        if model_dir.exists() {
            let model_path = model_dir.join("decoder_model.onnx");
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
            let model_path = hf_hub_download(&model_id, "decoder_model.onnx", None, None)?;
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
        environment: &'a Environment,
        model: &[u8],
        tokenizer_config: String,
        special_tokens_map: String,
        device: Device,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<Self> {
        let tokenizer = AutoTokenizer::new_from_memory(tokenizer_config, special_tokens_map)?;
        let model = ConditionalGenerationModel::new_from_memory(
            environment,
            model,
            device,
            optimization_level,
        )?;
        let token_type_support = model.get_token_type_support();

        Ok(ConditionalGenerationPipeline {
            tokenizer,
            model,
            token_type_support,
        })
    }

    /// Creates new pipeline from ONNX model file and tokenizer configuration.
    pub fn new_from_files(
        environment: &'a Environment,
        model: PathBuf,
        tokenizer_config: PathBuf,
        special_tokens_map: PathBuf,
        device: Device,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<Self> {
        let tokenizer = AutoTokenizer::new(tokenizer_config, special_tokens_map)?;
        let model = ConditionalGenerationModel::new_from_file(
            environment,
            model,
            device,
            optimization_level,
        )?;
        let token_type_support = model.get_token_type_support();

        Ok(ConditionalGenerationPipeline {
            tokenizer,
            model,
            token_type_support,
        })
    }

    pub fn generate<'sampler>(
        &self,
        prompt: &str,
        max_length: i32,
        sampler: &'sampler dyn Sampler,
    ) -> Result<String> {
        let prompt_batch = vec![prompt.to_string()];
        self.generate_batch(prompt_batch, max_length, sampler)
            .map(|batch| batch[0].clone())
    }

    /// Generates text from input batch text.
    pub fn generate_batch<'sampler>(
        &self,
        prompt: Vec<String>,
        max_length: i32,
        sampler: &'sampler dyn Sampler,
    ) -> Result<Vec<String>> {
        let batch_size = prompt.len();
        let encoding = self.tokenizer.tokenizer.encode_batch(prompt, false)?;
        let mut generated_ids = vec![vec![]; encoding.len()];
        let input_ids_tensors = &encoding
            .iter()
            .map(|e| Array1::from_iter(e.get_ids().iter().map(|i| *i as u32)))
            .collect::<Vec<Array1<u32>>>();

        let att_mask_tensors = &encoding
            .iter()
            .map(|e| Array1::from_iter(e.get_attention_mask().iter().map(|i| *i as u32)))
            .collect::<Vec<Array1<u32>>>();

        let type_ids_tensors = &encoding
            .iter()
            .map(|e| Array1::from_iter(e.get_type_ids().iter().map(|i| *i as u32)))
            .collect::<Vec<Array1<u32>>>();

        let mut input_ids = ndarray::stack(
            Axis(0),
            input_ids_tensors
                .iter()
                .map(|e| e.view())
                .collect::<Vec<ArrayView1<_>>>()
                .as_slice(),
        )?;
        let mut attention_mask = ndarray::stack(
            Axis(0),
            att_mask_tensors
                .iter()
                .map(|e| e.view())
                .collect::<Vec<ArrayView1<_>>>()
                .as_slice(),
        )?;
        let mut type_ids = ndarray::stack(
            Axis(0),
            type_ids_tensors
                .iter()
                .map(|e| e.view())
                .collect::<Vec<ArrayView1<_>>>()
                .as_slice(),
        )?;

        let mut eos_token_generated = vec![false; encoding.len()];
        for _ in 0..max_length {
            let output = self.model.forward(
                input_ids.to_owned(),
                Some(attention_mask.to_owned()),
                Some(type_ids.to_owned()),
            )?;
            let seq_len = output.shape()[1];
            let logits = output.index_axis_move(Axis(1), seq_len - 1);
            let next_tokens = Array::from_iter(sampler.sample(logits.view()));
            input_ids = concatenate(
                Axis(1),
                &[input_ids.view(), next_tokens.view().insert_axis(Axis(1))],
            )?;
            attention_mask = concatenate(
                Axis(1),
                &[attention_mask.view(), Array2::ones((batch_size, 1)).view()],
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
}

#[cfg(test)]
mod tests {
    use onnxruntime::environment::Environment;
    use onnxruntime::{GraphOptimizationLevel, LoggingLevel};

    use crate::common::Device;
    use crate::error::Result;
    use crate::sampling::TopKSampler;
    use crate::ConditionalGenerationPipeline;

    #[test]
    fn test_gen_model() -> Result<()> {
        let env = Environment::builder()
            .with_log_level(LoggingLevel::Error)
            .build()
            .unwrap();
        let pipeline = ConditionalGenerationPipeline::from_pretrained(
            &env,
            "optimum/gpt2".to_string(),
            Device::CPU,
            GraphOptimizationLevel::All,
        )?;
        let sampler = TopKSampler::new(5, 1.0);
        let output = pipeline.generate("Hello world mate", 10, &sampler)?;
        println!("{:?}", output);
        Ok(())
    }

    #[test]
    fn test_gen_model_batch() -> Result<()> {
        let env = Environment::builder()
            .with_log_level(LoggingLevel::Error)
            .build()
            .unwrap();
        let pipeline = ConditionalGenerationPipeline::from_pretrained(
            &env,
            "optimum/gpt2".to_string(),
            Device::CPU,
            GraphOptimizationLevel::All,
        )?;
        let sampler = TopKSampler::new(5, 1.0);
        let output = pipeline.generate_batch(
            vec![
                "Hello world mate".to_string(),
                "Hello world mate".to_string(),
            ],
            10,
            &sampler,
        )?;
        println!("{:?}", output);
        Ok(())
    }
}
