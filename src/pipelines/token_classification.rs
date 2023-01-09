use std::cell::{RefCell, RefMut};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::format;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use itertools::izip;
use more_asserts::assert_lt;
use onnxruntime::environment::Environment;
use onnxruntime::ndarray::{
    concatenate, s, Array, Array1, Array2, Array3, ArrayD, ArrayView1, Axis, Ix2, IxDyn,
};
use onnxruntime::session::{Input, Output, Session};
use onnxruntime::tensor::{FromArray, InputTensor, OrtOwnedTensor};
use onnxruntime::{ndarray, GraphOptimizationLevel};
use tokenizers::Offsets;

use crate::classification::ClassificationModel;
use crate::common::Device;
use crate::error::{Error, Result};
use crate::hf_hub::{get_ordered_labels_from_config, hf_hub_download};
use crate::modeling::conditional_generation::ConditionalGenerationModel;
use crate::sampling::Sampler;
use crate::tokenizer::AutoTokenizer;
use crate::{ClassPrediction, Embedding, EmbeddingModel, PoolingStrategy};

/// Wraps Huggingface Optimum pipeline exported to ONNX with `token-classification` task.
///
/// Export docs https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model
///
/// # Example
///
/// ```
/// use std::fs;
/// use edge_transformers::Device;
/// use onnxruntime::{GraphOptimizationLevel, LoggingLevel};
/// use onnxruntime::environment::Environment;
/// use edge_transformers::TokenClassificationPipeline;
///
/// let environment = Environment::builder()
///   .with_name("test")
///  .with_log_level(LoggingLevel::Verbose)
/// .build()
/// .unwrap();
///
/// let pipeline = TokenClassificationPipeline::from_pretrained(
///     &environment,
///     "optimum/bert-base-NER".to_string(),
///     Device::CPU,
///     GraphOptimizationLevel::All
/// ).unwrap();
///
/// let input = "This is a test";
///
/// println!("{:?}", pipeline.tag(input).unwrap());
/// ```
pub struct TokenClassificationPipeline<'a> {
    tokenizer: AutoTokenizer,
    model: ClassificationModel<'a>,
    labels: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TaggedString {
    pub input_string: String,
    pub tags: Vec<TokenClassPrediction>,
}

#[derive(Debug, Clone)]
pub struct TokenClassPrediction {
    pub best: ClassPrediction,
    pub all: Vec<ClassPrediction>,
    pub start: usize,
    pub end: usize,
}

impl<'a> TokenClassificationPipeline<'a> {
    pub fn from_pretrained(
        env: &'a Environment,
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
            let labels = match model_dir.join("config.json").to_str() {
                Some(path) => Some(get_ordered_labels_from_config(path)?),
                None => None,
            };
            Self::new_from_files(
                env,
                model_path,
                tokenizer_path,
                special_tokens_path,
                device,
                optimization_level,
                labels,
            )
        } else {
            let model_path = hf_hub_download(&model_id, "model.onnx", None, None)?;
            let tokenizer_path = hf_hub_download(&model_id, "tokenizer.json", None, None)?;
            let mut special_tokens_path =
                hf_hub_download(&model_id, "special_tokens_map.json", None, None);
            if special_tokens_path.is_err() {
                special_tokens_path = hf_hub_download(&model_id, "config.json", None, None);
            }
            let labels = match hf_hub_download(&model_id, "config.json", None, None) {
                Ok(labels) => match get_ordered_labels_from_config(labels.to_str().unwrap()) {
                    Ok(labels) => Some(labels),
                    Err(_) => None,
                },
                Err(_) => None,
            };
            Self::new_from_files(
                env,
                model_path,
                tokenizer_path,
                special_tokens_path?,
                device,
                optimization_level,
                labels,
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
    /// * `special_tokens_map` - Path to special tokens map file. Maps token names to their string values.
    /// * `device` - Device to run the model on.
    /// * `optimization_level` - ONNX Runtime graph optimization level.
    pub fn new_from_files(
        environment: &'a Environment,
        model_path: PathBuf,
        tokenizer_config: PathBuf,
        special_tokens_map: PathBuf,
        device: Device,
        optimization_level: GraphOptimizationLevel,
        labels: Option<Vec<String>>,
    ) -> Result<Self> {
        let tokenizer = AutoTokenizer::new(tokenizer_config, special_tokens_map)?;
        let model = ClassificationModel::new_from_file(
            environment,
            model_path,
            device,
            optimization_level,
        )?;
        if !model.is_token_classification() {
            return Err(Error::GenericError {
                message:
                    "ONNX Model is sequence classification model, not token classification model"
                        .to_string(),
            });
        }
        let labels = match labels {
            Some(labels) => {
                if labels.len() != model.get_num_labels() {
                    return Err(Error::GenericError { message: format!("Number of labels in model ({}) does not match number of labels provided ({})", model.get_num_labels(), labels.len()) } );
                }
                labels
            }
            None => {
                let labels = model.get_num_labels();
                (0..labels)
                    .map(|i| format!("LABEL_{}", i.to_string()))
                    .collect()
            }
        };
        Ok(Self {
            tokenizer,
            model,
            labels,
        })
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
        device: Device,
        optimization_level: GraphOptimizationLevel,
        labels: Option<Vec<String>>,
    ) -> Result<Self> {
        let tokenizer = AutoTokenizer::new_from_memory(tokenizer_config, special_tokens_map)?;
        let model =
            ClassificationModel::new_from_memory(environment, model, device, optimization_level)?;

        if !model.is_token_classification() {
            return Err(Error::GenericError {
                message:
                    "ONNX Model is sequence classification model, not token classification model"
                        .to_string(),
            });
        }
        let labels = match labels {
            Some(labels) => {
                if labels.len() != model.get_num_labels() {
                    return Err(Error::GenericError { message: format!("Number of labels in model ({}) does not match number of labels provided ({})", model.get_num_labels(), labels.len()) } );
                }
                labels
            }
            None => {
                let labels = model.get_num_labels();
                (0..labels)
                    .map(|i| format!("LABEL_{}", i.to_string()))
                    .collect()
            }
        };
        Ok(Self {
            tokenizer,
            model,
            labels,
        })
    }

    /// Embeds input text.
    ///
    /// # Arguments
    ///
    /// * `input` - Input text.
    pub fn tag(&self, input: &str) -> Result<TaggedString> {
        let tokenized = self.tokenizer.tokenizer.encode(input, false)?;
        let input_ids = Array1::from_iter(tokenized.get_ids().iter().map(|i| *i as u32));
        let input_ids = input_ids.insert_axis(Axis(0));
        let attention_mask =
            Array1::from_iter(tokenized.get_attention_mask().iter().map(|i| *i as u32));
        let attention_mask = attention_mask.insert_axis(Axis(0));
        let token_type_ids = Array1::from_iter(tokenized.get_type_ids().iter().map(|i| *i as u32));
        let token_type_ids = token_type_ids.insert_axis(Axis(0));
        let offsets = tokenized.get_offsets();

        let mut scores =
            self.model
                .forward(input_ids, attention_mask.to_owned(), Some(token_type_ids))?;

        let mut output = self.scores_to_tagged_strings(
            vec![input.to_string()],
            scores.into_dimensionality()?,
            attention_mask,
            vec![offsets],
        );
        Ok(output.pop().unwrap())
    }

    /// Embeds input texts.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input texts.
    pub fn tag_batch(&self, inputs: Vec<String>) -> Result<Vec<TaggedString>> {
        let tokenized = self
            .tokenizer
            .tokenizer
            .encode_batch(inputs.clone(), false)?;
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
        let offsets = tokenized
            .iter()
            .map(|t| t.get_offsets())
            .collect::<Vec<_>>();

        let output =
            self.model
                .forward(input_ids, attention_mask.to_owned(), Some(token_type_ids))?;

        let output = self.scores_to_tagged_strings(
            inputs,
            output.into_dimensionality()?,
            attention_mask,
            offsets,
        );
        Ok(output)
    }

    fn scores_to_tagged_strings(
        &self,
        original_strings: Vec<String>,
        scores: Array3<f32>,
        attention_mask: Array2<u32>,
        offsets: Vec<&[Offsets]>,
    ) -> Vec<TaggedString> {
        let mut predictions = Vec::new();
        for (original_string, scores, attention_mask, offsets) in izip!(
            original_strings,
            scores.outer_iter(),
            attention_mask.outer_iter(),
            offsets
        ) {
            let mut prediction = TaggedString {
                input_string: original_string.to_string(),
                tags: vec![],
            };
            for (score, attention_mask, offsets) in
                izip!(scores.outer_iter(), attention_mask, offsets)
            {
                if *attention_mask == 0 {
                    continue;
                }
                let mut max_score = f32::NEG_INFINITY;
                let mut max_index = 0;
                let mut all_classes: Vec<ClassPrediction> = vec![];
                for (i, (score)) in izip!(score).enumerate() {
                    if *score > max_score {
                        max_score = *score;
                        max_index = i;
                    }
                    all_classes.push(ClassPrediction {
                        label: self.labels[i].clone(),
                        score: *score,
                    });
                }
                prediction.tags.push(TokenClassPrediction {
                    start: offsets.0,
                    end: offsets.1,
                    best: ClassPrediction {
                        label: self.labels[max_index].clone(),
                        score: max_score,
                    },
                    all: all_classes,
                });
            }
            predictions.push(prediction);
        }
        predictions
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use onnxruntime::LoggingLevel;

    use super::*;

    #[test]
    fn test_tok_classification_pipeline() {
        let environment = Environment::builder()
            .with_name("embedding_pipeline")
            .with_log_level(LoggingLevel::Verbose)
            .build()
            .unwrap();
        let pipeline = TokenClassificationPipeline::from_pretrained(
            &environment,
            "optimum/bert-base-NER".to_string(),
            Device::CPU,
            GraphOptimizationLevel::All,
        )
        .unwrap();

        let input = "This is a test";

        let output = pipeline.tag(input).unwrap();

        assert_eq!(output.input_string, input);

        assert_eq!(output.tags.len(), 4);
        assert_eq!(output.tags[0].start, 0);
        assert_eq!(output.tags[0].end, 4);
        assert_lt!(output.tags[0].start, output.tags[1].start);
        assert_lt!(output.tags[0].end, output.tags[1].end);
        assert!(output.tags[0].best.score > 0.0);
        assert!(output.tags[0].all[0].score > 0.0);
        assert!(output.tags[0].all[1].score > 0.0);
        assert!(output.tags[0].best.score >= output.tags[0].all[0].score);
        assert!(output.tags[0].best.score >= output.tags[0].all[1].score);
    }
}
