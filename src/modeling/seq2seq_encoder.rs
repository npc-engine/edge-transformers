use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use ndarray::{Array, Array2, Array3, IxDyn};
use ort::environment::Environment;
use ort::session::Session;
use ort::tensor::{FromArray, InputTensor};
use ort::{GraphOptimizationLevel, InMemorySession, SessionBuilder};

use crate::common::Device;
use crate::common::{apply_device, match_to_inputs};
use crate::error::Result;
use crate::ORTSession;

/// Onnx inference session wrapper for the conditional generation models.
pub struct Seq2SeqEncoderModel<'a> {
    model_session: ORTSession<'a>,
    token_type_support: bool,
}

impl<'a> Seq2SeqEncoderModel<'a> {
    pub fn new_from_memory(
        env: Arc<Environment>,
        model_bytes: &'a [u8],
        device: Device,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<Self> {
        let mut session_builder = SessionBuilder::new(&env)?;

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
        })
    }

    pub fn new_from_file(
        env: Arc<Environment>,
        model_path: PathBuf,
        device: Device,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<Self> {
        let mut session_builder = SessionBuilder::new(&env)?;
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
    ) -> Result<Array3<f32>> {
        let input_map = self.prepare_input_map(input_ids, attention_mask, token_type_ids)?;
        let mut model = match &self.model_session {
            ORTSession::Owned(m) => m,
            ORTSession::InMemory(m) => m,
        };
        let input_tensor = match_to_inputs(&model.inputs, input_map)?;
        let output_names = model
            .outputs
            .iter()
            .map(|o| o.name.clone())
            .collect::<Vec<_>>();
        let outputs_tensors = model.run(input_tensor)?;
        let output_map: HashMap<String, Array<f32, IxDyn>> = output_names
            .iter()
            .map(|name| name.to_string())
            .zip(
                outputs_tensors
                    .into_iter()
                    .map(|tensor| tensor.try_extract().unwrap().view().to_owned()),
            )
            .collect();
        Ok(output_map
            .get("last_hidden_state")
            .expect("Model must have last_hidden_state output")
            .to_owned()
            .into_dimensionality()?)
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

#[cfg(test)]
mod tests {
    use crate::hf_hub::hf_hub_download;

    use super::*;

    #[test]
    fn test_encoder() -> Result<()> {
        let env = Environment::builder().build().unwrap();
        let model = Seq2SeqEncoderModel::new_from_file(
            env.into_arc(),
            hf_hub_download("optimum/t5-small", "encoder_model.onnx", None, None).unwrap(),
            Device::CPU,
            GraphOptimizationLevel::Disable,
        )
        .unwrap();
        let input_ids1 =
            Array2::from_shape_vec((1, 8), vec![101, 2000, 1037, 1037, 1037, 1037, 1037, 102])
                .unwrap();
        let attention_mask = Array2::from_shape_vec((1, 8), vec![1, 1, 1, 1, 1, 1, 1, 1]).unwrap();
        let embeddings = model
            .forward(input_ids1, Some(attention_mask), None)
            .unwrap();
        assert_eq!(*embeddings.shape().get(0).unwrap(), 1 as usize);
        Ok(())
    }
}
