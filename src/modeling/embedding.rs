use std::cell::{RefCell, RefMut};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use onnxruntime::environment::Environment;
use onnxruntime::ndarray::{s, Array, Array1, Array2, ArrayD, Axis, Ix1, Ix2, IxDyn};
use onnxruntime::session::{Input, Output, Session};
use onnxruntime::tensor::{FromArray, InputTensor, OrtOwnedTensor};
use onnxruntime::GraphOptimizationLevel;

use crate::common::Device;
use crate::common::{apply_device, match_to_inputs};
use crate::error::{Error, Result};
use crate::sampling::Sampler;

pub enum PoolingStrategy {
    Mean,
    Max,
    First,
}

/// Onnx inference session wrapper for the conditional generation models.
pub struct EmbeddingModel<'a> {
    model_session: RefCell<Session<'a>>,
    token_type_support: bool,
    pub pooling: PoolingStrategy,
}

/// Holds text embedding with model .
pub struct Embedding {
    pub embedding: Array1<f32>,
}

impl<'a> EmbeddingModel<'a> {
    pub fn new_from_memory(
        env: &'a Environment,
        model_bytes: &[u8],
        pooling: PoolingStrategy,
        device: Device,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<Self> {
        let mut session_builder = env.new_session_builder()?;

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
            model_session: RefCell::new(session),
            token_type_support,
            pooling,
        })
    }

    pub fn new_from_file<'path>(
        env: &'a Environment,
        model_path: PathBuf,
        pooling: PoolingStrategy,
        device: Device,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<Self> {
        let mut session_builder = env.new_session_builder()?;
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
            model_session: RefCell::new(session),
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
        let mut input_tensor = match_to_inputs(&self.model_session.borrow().inputs, input_map)?;
        let mut model = self.model_session.borrow_mut();
        let output_names = model
            .outputs
            .iter()
            .map(|o| o.name.clone())
            .collect::<Vec<_>>();
        let outputs_tensors = model.run(input_tensor)?;
        let mut output_map: HashMap<String, Array<f32, IxDyn>> = output_names
            .iter()
            .map(|name| name.to_string())
            .zip(outputs_tensors.into_iter().map(|tensor| {
                Array::<f32, IxDyn>::from_shape_vec(
                    tensor.shape(),
                    tensor.iter().map(|x| *x).collect(),
                )
                .unwrap()
            }))
            .collect();
        // Use last_hidden_state as embedding and compute pooling on it
        let embeddings = match self.pooling {
            PoolingStrategy::Mean => {
                let last_hidden_state = output_map.get("last_hidden_state").unwrap();
                let mut embeddings = last_hidden_state.mean_axis(Axis(1)).unwrap();
                embeddings
            }
            PoolingStrategy::Max => {
                let mut last_hidden_state = output_map.get_mut("last_hidden_state").unwrap();
                last_hidden_state.accumulate_axis_inplace(Axis(1), |a, b| *b = a.max(*b));
                last_hidden_state.to_owned()
            }
            PoolingStrategy::First => {
                let last_hidden_state = output_map.get("last_hidden_state").unwrap();
                let mut embeddings = last_hidden_state.slice_axis(Axis(1), (0..1).into());
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
    ) -> Result<HashMap<String, InputTensor<IxDyn>>> {
        let mut input_map = HashMap::<String, InputTensor<IxDyn>>::new();
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
            &env,
            hf_hub_download("optimum/all-MiniLM-L6-v2", "model.onnx", None, None).unwrap(),
            PoolingStrategy::Mean,
            Device::CPU,
            GraphOptimizationLevel::DisableAll,
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
