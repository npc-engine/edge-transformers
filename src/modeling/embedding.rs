use half::{bf16, f16};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use ndarray::{Array, Array1, Array2, Axis, IxDyn};
use ort::environment::Environment;
use ort::tensor::{FromArray, InputTensor, TensorElementDataType};
use ort::{GraphOptimizationLevel, InMemorySession, Session, SessionBuilder};

use crate::common::Device;
use crate::common::{apply_device, match_to_inputs};
use crate::error::Result;
use crate::ORTSession;

pub enum PoolingStrategy {
    Mean,
    Max,
    First,
}

/// Onnx inference session wrapper for the conditional generation models.
pub struct EmbeddingModel<'a> {
    model_session: ORTSession<'a>,
    token_type_support: bool,
    pub pooling: PoolingStrategy,
}

/// Holds text embedding with model .
pub struct Embedding {
    pub embedding: Array1<f32>,
}

impl<'a> EmbeddingModel<'a> {
    pub fn new_from_memory(
        env: &'a Arc<Environment>,
        model_bytes: &'a [u8],
        pooling: PoolingStrategy,
        device: Device,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<Self> {
        let mut session_builder = SessionBuilder::new(env)?;

        session_builder = apply_device(session_builder, device)?;
        let session = session_builder
            .with_optimization_level(optimization_level)?
            .with_model_from_memory(model_bytes)?;
        let token_type_support = session.inputs.len() == 3
            && session
                .inputs
                .iter()
                .filter(|i| i.name == "token_type_ids")
                .count()
                > 0;
        Ok(Self {
            model_session: ORTSession::InMemory(session),
            token_type_support,
            pooling,
        })
    }

    pub fn new_from_file(
        env: Arc<Environment>,
        model_path: PathBuf,
        pooling: PoolingStrategy,
        device: Device,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<Self> {
        let mut session_builder = SessionBuilder::new(&env)?;
        session_builder = apply_device(session_builder, device)?;
        let session = session_builder
            .with_optimization_level(optimization_level)?
            .with_model_from_file(model_path)?;
        let token_type_support = session.inputs.len() == 3
            && session
                .inputs
                .iter()
                .filter(|i| i.name == "token_type_ids")
                .count()
                > 0;
        Ok(Self {
            model_session: ORTSession::Owned(session),
            token_type_support,
            pooling,
        })
    }

    pub fn get_token_type_support(&self) -> bool {
        self.token_type_support
    }

    /// Does inference
    ///
    /// Returns a vector of Embedding class that contains the embedding.
    pub fn forward(
        &self,
        input_ids: Array2<u32>,
        attention_mask: Option<Array2<u32>>,
        token_type_ids: Option<Array2<u32>>,
    ) -> Result<Vec<Embedding>> {
        let input_map = self.prepare_input_map(input_ids, attention_mask, token_type_ids)?;
        let model = match &self.model_session {
            ORTSession::Owned(s) => s,
            ORTSession::InMemory(s) => s,
        };
        let input_tensor = match_to_inputs(&model.inputs, input_map)?;
        let output_names = model
            .outputs
            .iter()
            .map(|o| o.name.clone())
            .collect::<Vec<_>>();
        let outputs_tensors = model.run(input_tensor)?;

        let mut output_map = HashMap::new();
        for (name, tensor) in output_names.iter().zip(outputs_tensors) {
            let extracted = match tensor.data_type() {
                TensorElementDataType::Float16 => tensor
                    .try_extract::<f16>()?
                    .view()
                    .to_owned()
                    .mapv(|v| v.to_f32()),
                TensorElementDataType::Float32 => tensor.try_extract::<f32>()?.view().to_owned(),
                TensorElementDataType::Float64 => tensor
                    .try_extract::<f64>()?
                    .view()
                    .to_owned()
                    .mapv(|v| v as f32),
                TensorElementDataType::Bfloat16 => tensor
                    .try_extract::<bf16>()?
                    .view()
                    .to_owned()
                    .mapv(|v| v.to_f32()),
                _ => {
                    return Err(
                        format!("Unsupported output data type {:?}", tensor.data_type()).into(),
                    )
                }
            };
            let dimensionality = extracted.into_dimensionality::<IxDyn>()?;
            output_map.insert(name.to_string(), dimensionality);
        }
        // Use last_hidden_state as embedding and compute pooling on it
        let embeddings = match self.pooling {
            PoolingStrategy::Mean => {
                let last_hidden_state = output_map.get("last_hidden_state").unwrap();
                let embeddings = last_hidden_state.mean_axis(Axis(1)).unwrap();
                embeddings
            }
            PoolingStrategy::Max => {
                let last_hidden_state = output_map.get_mut("last_hidden_state").unwrap();
                last_hidden_state.accumulate_axis_inplace(Axis(1), |a, b| *b = a.max(*b));
                last_hidden_state.to_owned()
            }
            PoolingStrategy::First => {
                let last_hidden_state = output_map.get("last_hidden_state").unwrap();
                let embeddings = last_hidden_state.slice_axis(Axis(1), (0..1).into());
                embeddings.to_owned()
            }
        };
        let embeddings = embeddings
            .axis_iter(Axis(0))
            .map(|embedding| {
                Ok(Embedding {
                    embedding: embedding.to_owned().into_dimensionality()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(embeddings)
    }

    fn prepare_input_map(
        &self,
        input_ids: Array2<u32>,
        attention_mask: Option<Array2<u32>>,
        token_type_ids: Option<Array2<u32>>,
    ) -> Result<HashMap<String, InputTensor>> {
        let mut input_map = HashMap::<String, InputTensor>::new();
        let attention_mask = if attention_mask.is_none() {
            Array::ones((input_ids.shape()[0], input_ids.shape()[1]))
        } else {
            attention_mask.unwrap()
        };
        if self.token_type_support {
            if let Some(token_types_array) = token_type_ids {
                input_map.insert(
                    "token_type_ids".to_string(),
                    InputTensor::from_array(token_types_array.into_dimensionality()?),
                );
            } else {
                input_map.insert(
                    "token_type_ids".to_string(),
                    InputTensor::from_array(
                        Array2::<u32>::zeros((input_ids.nrows(), input_ids.ncols()))
                            .into_dimensionality()?,
                    ),
                );
            }
        }
        input_map.insert(
            "input_ids".to_string(),
            InputTensor::from_array(input_ids.into_dimensionality()?),
        );
        input_map.insert(
            "attention_mask".to_string(),
            InputTensor::from_array(attention_mask.into_dimensionality()?),
        );
        Ok(input_map)
    }
}

impl Embedding {
    pub fn new(embedding: Array1<f32>) -> Self {
        Embedding { embedding }
    }

    pub fn dot(&self, other: &Embedding) -> f32 {
        self.embedding.dot(&other.embedding)
    }

    /// Compute cosine similarity to other embedding
    pub fn similarity(&self, other: &Self) -> f32 {
        self.embedding.dot(&other.embedding)
            / (self
                .embedding
                .iter()
                .map(|x| x.powf(2f32))
                .sum::<f32>()
                .sqrt()
                * other
                    .embedding
                    .iter()
                    .map(|x| x.powf(2f32))
                    .sum::<f32>()
                    .sqrt())
    }
}

#[cfg(test)]
mod tests {
    use crate::hf_hub::hf_hub_download;

    use super::*;

    #[test]
    fn test_bert() {
        let env = Environment::builder().build().unwrap();
        let bert = EmbeddingModel::new_from_file(
            env.into_arc(),
            hf_hub_download("optimum/all-MiniLM-L6-v2", "model.onnx", None, None).unwrap(),
            PoolingStrategy::Mean,
            Device::CPU,
            GraphOptimizationLevel::Disable,
        )
        .unwrap();
        let input_ids1 =
            Array2::from_shape_vec((1, 8), vec![101, 2000, 1037, 1037, 1037, 1037, 1037, 102])
                .unwrap();
        let input_ids2 =
            Array2::from_shape_vec((1, 8), vec![101, 2000, 1037, 1037, 1037, 1037, 1037, 102])
                .unwrap();
        let attention_mask = Array2::from_shape_vec((1, 8), vec![1, 1, 1, 1, 1, 1, 1, 1]).unwrap();
        let embeddings = bert
            .forward(input_ids1, Some(attention_mask), None)
            .unwrap();
        let attention_mask = Array2::from_shape_vec((1, 8), vec![1, 1, 1, 1, 1, 1, 1, 1]).unwrap();
        let embeddings2 = bert
            .forward(input_ids2, Some(attention_mask), None)
            .unwrap();
        assert_eq!(embeddings.len(), 1);
        assert!(embeddings[0].similarity(&embeddings2[0]) > 0.9);
    }
}
