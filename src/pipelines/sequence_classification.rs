use std::path::{Path, PathBuf};

use onnxruntime::environment::Environment;
use onnxruntime::ndarray::{Array1, Array2, Axis};
use onnxruntime::GraphOptimizationLevel;

use crate::classification::ClassificationModel;
use crate::common::Device;
use crate::error::{Error, Result};
use crate::hf_hub::{get_ordered_labels_from_config, hf_hub_download};
use crate::tokenizer::AutoTokenizer;

/// Wraps Huggingface Optimum pipeline exported to ONNX with `sequence-classification` task.
///
/// Export docs https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model
///
/// # Example
///
/// ```
/// use std::fs;
/// use onnxruntime::{GraphOptimizationLevel, LoggingLevel};
/// use onnxruntime::environment::Environment;
/// use edge_transformers::{EmbeddingPipeline, PoolingStrategy, SequenceClassificationPipeline, Device};
///
/// let environment = Environment::builder()
///   .with_name("test")
///  .with_log_level(LoggingLevel::Verbose)
/// .build()
/// .unwrap();
///
/// let pipeline = SequenceClassificationPipeline::from_pretrained(
///     &environment,
///    "npc-engine/deberta-v3-small-finetuned-hate_speech18".to_string(),
///     Device::CPU,
///     GraphOptimizationLevel::All,
/// ).unwrap();
///
/// let input = "This is a test";
///
/// println!("Best label {:?}", pipeline.classify(input).unwrap().best.label);
/// ```
pub struct SequenceClassificationPipeline<'a> {
    tokenizer: AutoTokenizer,
    model: ClassificationModel<'a>,
    labels: Vec<String>,
}

pub struct Prediction {
    pub best: ClassPrediction,
    pub all: Vec<ClassPrediction>,
}

#[derive(Debug, Clone)]
pub struct ClassPrediction {
    pub label: String,
    pub score: f32,
}

impl<'a> SequenceClassificationPipeline<'a> {
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
        if model.is_token_classification() {
            return Err(Error::GenericError {
                message:
                    "ONNX Model is token classification model, not sequence classification model"
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

        if model.is_token_classification() {
            return Err(Error::GenericError {
                message:
                    "ONNX Model is token classification model, not sequence classification model"
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
    pub fn classify(&self, input: &str) -> Result<Prediction> {
        let tokenized = self.tokenizer.tokenizer.encode(input, false)?;
        let input_ids = Array1::from_iter(tokenized.get_ids().iter().map(|i| *i as u32));
        let input_ids = input_ids.insert_axis(Axis(0));
        let attention_mask =
            Array1::from_iter(tokenized.get_attention_mask().iter().map(|i| *i as u32));
        let attention_mask = attention_mask.insert_axis(Axis(0));
        let token_type_ids = Array1::from_iter(tokenized.get_type_ids().iter().map(|i| *i as u32));
        let token_type_ids = token_type_ids.insert_axis(Axis(0));

        let scores = self
            .model
            .forward(input_ids, attention_mask, Some(token_type_ids))?;

        let mut output = self.scores_to_predictions(scores.into_dimensionality()?);

        Ok(output.pop().unwrap())
    }

    /// Embeds input texts.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input texts.
    pub fn classify_batch(&self, inputs: Vec<String>) -> Result<Vec<Prediction>> {
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

        let output = self
            .model
            .forward(input_ids, attention_mask, Some(token_type_ids))?;
        let output = self.scores_to_predictions(output.into_dimensionality()?);
        Ok(output)
    }

    fn scores_to_predictions(&self, scores: Array2<f32>) -> Vec<Prediction> {
        let mut predictions = Vec::new();
        for i in 0..scores.shape()[0] {
            let mut class_predictions = Vec::new();
            for j in 0..scores.shape()[1] {
                class_predictions.push(ClassPrediction {
                    label: self.labels[j].clone(),
                    score: scores[[i, j]],
                });
            }
            let best_prediction = class_predictions
                .iter()
                .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
                .unwrap();
            predictions.push(Prediction {
                best: best_prediction.clone(),
                all: class_predictions,
            });
        }
        predictions
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
        let pipeline = SequenceClassificationPipeline::from_pretrained(
            &environment,
            "npc-engine/deberta-v3-small-finetuned-hate_speech18".to_string(),
            Device::CPU,
            GraphOptimizationLevel::All,
        )
        .unwrap();

        let input = "This is a test";

        let output = pipeline.classify(input).unwrap();

        assert!(output.best.score > 0.0);
        assert!(output.best.score < 1.0);
        let mut sum: f32 = 0.0;
        for class_prediction in output.all {
            assert!(class_prediction.score > 0.0);
            assert!(class_prediction.score < 1.0);
            sum += class_prediction.score;
        }
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
