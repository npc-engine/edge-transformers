use std::path::{Path, PathBuf};

use onnxruntime::GraphOptimizationLevel;
use onnxruntime::environment::Environment;
use onnxruntime::ndarray::{
    Array1, Array2, Axis,
};

use crate::{Embedding, EmbeddingModel, PoolingStrategy};
use crate::common::Device;
use crate::error::Result;
use crate::hf_hub::hf_hub_download;
use crate::tokenizer::AutoTokenizer;

/// Wraps Huggingface Optimum pipeline exported to ONNX with `default` task.
///
///
/// Export docs https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model
///
/// # Example
///
/// ```
/// use std::fs;
/// use onnxruntime::environment::Environment;
/// use onnxruntime::{GraphOptimizationLevel, LoggingLevel};
/// use edge_transformers::{EmbeddingPipeline, PoolingStrategy,Device};
///
/// let environment = Environment::builder()
///   .with_name("test")
///  .with_log_level(LoggingLevel::Verbose)
/// .build()
/// .unwrap();
///
/// let pipeline = EmbeddingPipeline::from_pretrained(
///  &environment,
/// "optimum/all-MiniLM-L6-v2".to_string(),
/// PoolingStrategy::Mean,
/// Device::CPU,
/// GraphOptimizationLevel::All,
/// ).unwrap();
///
/// let input = "This is a test";
/// let emb1 = pipeline.embed(input).unwrap();
/// let input = "This is a test2";
/// let emb2 = pipeline.embed(input).unwrap();
/// println!("Similarity: {:?}", emb1.similarity(&emb2));
/// ```
pub struct EmbeddingPipeline<'a> {
    tokenizer: AutoTokenizer,
    model: EmbeddingModel<'a>,
}

impl<'a> EmbeddingPipeline<'a> {
    pub fn from_pretrained(
        env: &'a Environment,
        model_id: String,
        pool_strategy: PoolingStrategy,
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
                pool_strategy,
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
                pool_strategy,
                device,
                optimization_level,
            )
        }
    }

    /// Creates new pipeline from model and tokenizer configuration files.
    ///
    /// # Arguments
    ///
    /// * `environment` - ONNX Runtime environment.
    /// * `model_path` - Path to ONNX model file.
    /// * `tokenizer_config` - Path to tokenizer configuration file.
    /// * `special_tokens_map` - Path to special tokens map file.
    /// * `device` - Device to run the model on.
    /// * `optimization_level` - ONNX Runtime graph optimization level.
    pub fn new_from_files(
        environment: &'a Environment,
        model_path: PathBuf,
        tokenizer_config: PathBuf,
        special_tokens_map: PathBuf,
        pooling: PoolingStrategy,
        device: Device,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<Self> {
        let tokenizer = AutoTokenizer::new(tokenizer_config, special_tokens_map)?;
        let model = EmbeddingModel::new_from_file(
            environment,
            model_path,
            pooling,
            device,
            optimization_level,
        )?;
        Ok(Self { tokenizer, model })
    }

    /// Creates new pipeline from model and tokenizer configuration files.
    ///
    /// # Arguments
    ///
    /// * `environment` - ONNX Runtime environment.
    /// * `model` - ONNX model file content.
    /// * `tokenizer_config` - Path to tokenizer configuration file.
    /// * `special_tokens_map` - Path to special tokens map file.
    /// * `device` - Device to run the model on.
    /// * `optimization_level` - ONNX Runtime graph optimization level.
    pub fn new_from_memory(
        environment: &'a Environment,
        model: &[u8],
        tokenizer_config: String,
        special_tokens_map: String,
        pooling: PoolingStrategy,
        device: Device,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<Self> {
        let tokenizer = AutoTokenizer::new_from_memory(tokenizer_config, special_tokens_map)?;
        let model = EmbeddingModel::new_from_memory(
            environment,
            model,
            pooling,
            device,
            optimization_level,
        )?;
        Ok(Self { tokenizer, model })
    }

    /// Embeds input text.
    ///
    /// # Arguments
    ///
    /// * `input` - Input text.
    pub fn embed(&self, input: &str) -> Result<Embedding> {
        let tokenized = self.tokenizer.tokenizer.encode(input, false)?;
        let input_ids = Array1::from_iter(tokenized.get_ids().iter().map(|i| *i as u32));
        let input_ids = input_ids.insert_axis(Axis(0));
        let attention_mask =
            Array1::from_iter(tokenized.get_attention_mask().iter().map(|i| *i as u32));
        let attention_mask = attention_mask.insert_axis(Axis(0));
        let token_type_ids = Array1::from_iter(tokenized.get_type_ids().iter().map(|i| *i as u32));
        let token_type_ids = token_type_ids.insert_axis(Axis(0));

        let mut output =
            self.model
                .forward(input_ids, Some(attention_mask), Some(token_type_ids))?;
        Ok(output.pop().unwrap())
    }

    /// Embeds input texts.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input texts.
    pub fn embed_batch(&self, inputs: Vec<String>) -> Result<Vec<Embedding>> {
        let tokenized = self.tokenizer.tokenizer.encode_batch(inputs, false)?;
        let input_ids = tokenized.iter().map(|t| t.get_ids()).collect::<Vec<_>>();
        let input_ids =
            Array2::from_shape_vec((input_ids.len(), input_ids[0].len()), input_ids.concat())?;
        let attention_mask = tokenized
            .iter()
            .map(|t| t.get_attention_mask())
            .collect::<Vec<_>>();
        let attention_mask = Array2::from_shape_vec(
            (attention_mask.len(), attention_mask[0].len()),
            attention_mask.concat(),
        )?;
        let token_type_ids = tokenized
            .iter()
            .map(|t| t.get_type_ids())
            .collect::<Vec<_>>();
        let token_type_ids = Array2::from_shape_vec(
            (token_type_ids.len(), token_type_ids[0].len()),
            token_type_ids.concat(),
        )?;

        let output =
            self.model
                .forward(input_ids, Some(attention_mask), Some(token_type_ids))?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use onnxruntime::LoggingLevel;

    use super::*;

    #[test]
    fn test_embedding_pipeline() {
        let environment = Environment::builder()
            .with_name("embedding_pipeline")
            .with_log_level(LoggingLevel::Verbose)
            .build()
            .unwrap();
        let pipeline = EmbeddingPipeline::from_pretrained(
            &environment,
            "optimum/all-MiniLM-L6-v2".to_string(),
            PoolingStrategy::Mean,
            Device::CPU,
            GraphOptimizationLevel::All,
        )
        .unwrap();

        let input = "This is a test";
        let input1 = "This is a test";

        let embedding = pipeline.embed(input).unwrap();
        let embeddings = pipeline
            .embed_batch(vec![input.to_string(), input1.to_string()])
            .unwrap();

        let sim1 = embedding.similarity(&embeddings[0]);
        let sim2 = embedding.similarity(&embeddings[1]);

        assert!(sim1 > -1.0);
        assert!(sim2 > -1.0);
        assert!(sim1 < 1.0);
        assert!(sim2 < 1.0);
    }
}
