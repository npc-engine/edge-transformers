use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use ndarray::{Array, Array2, Array3, IxDyn};
use ort::environment::Environment;
use ort::session::{Input, Output, Session};
use ort::tensor::{FromArray, InputTensor};
use ort::{GraphOptimizationLevel, InMemorySession, SessionBuilder};

use crate::common::Device;
use crate::common::{apply_device, match_to_inputs};
use crate::error::{Error, Result};
use crate::ORTSession;

/// Onnx inference session wrapper for the conditional generation models.
///
/// Validates inputs and outputs of the model and provides a convenient interface to the model.
pub struct Seq2SeqDecoderModel<'a> {
    model_session: ORTSession<'a>,
    token_type_support: bool,
}

impl<'a> Seq2SeqDecoderModel<'a> {
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
        let token_type_support = Self::validate_signature(&session.inputs, &session.outputs)?;
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
        session_builder = apply_device(session_builder, device)?;
        let session = session_builder
            .with_optimization_level(optimization_level)?
            .with_model_from_file(model_path)?;
        let token_type_support = Self::validate_signature(&session.inputs, &session.outputs)?;
        Ok(Self {
            model_session: ORTSession::Owned(session),
            token_type_support,
        })
    }

    fn validate_signature(inputs: &Vec<Input>, outputs: &Vec<Output>) -> Result<bool> {
        let token_type_support = inputs.iter().any(|input| input.name == "token_type_ids");
        let past_values: Vec<String> = inputs
            .iter()
            .filter(|input| input.name.contains("past_key_values"))
            .map(|input| input.name.to_string())
            .collect();
        if past_values.len() != 0 {
            return Err(Error::OnnxIncorrectInputs {
                message:
                    "The model supports past key values. Use Seq2SeqDecoderModelWithPKVs instead"
                        .to_string(),
                expected: vec!["input_ids".to_string(), "attention_mask".to_string()],
                actual: past_values,
            });
        }
        if inputs
            .iter()
            .all(|inp| inp.name != "encoder_attention_mask")
            || inputs.iter().all(|inp| inp.name != "input_ids")
            || inputs.iter().all(|inp| inp.name != "encoder_hidden_states")
        {
            return Err(Error::OnnxIncorrectInputs {
                message: "The model does not have the required inputs.".to_string(),
                actual: inputs.iter().map(|inp| inp.name.to_string()).collect(),
                expected: vec![
                    "encoder_attention_mask".to_string(),
                    "input_ids".to_string(),
                    "encoder_hidden_states".to_string(),
                ],
            });
        }
        if outputs.iter().all(|output| output.name != "logits") {
            return Err(Error::OnnxIncorrectOutputs {
                message: "The model does not have a logits output.".to_string(),
                actual: outputs.iter().map(|inp| inp.name.to_string()).collect(),
                expected: vec!["logits".to_string()],
            });
        }
        Ok(token_type_support)
    }

    /// Does inference.
    pub fn forward(
        &self,
        input_ids: Array2<u32>,
        encoder_last_hidden_state: Array3<f32>,
        encoder_attention_mask: Option<Array2<u32>>,
    ) -> Result<Array3<f32>> {
        let input_map =
            self.prepare_input_map(input_ids, encoder_last_hidden_state, encoder_attention_mask)?;
        let mut model = match &self.model_session {
            ORTSession::Owned(model) => model,
            ORTSession::InMemory(model) => model,
        };
        let input_tensor = match_to_inputs(&model.inputs, input_map)?;
        let output_names: Vec<String> = model
            .outputs
            .iter()
            .map(|output| output.name.to_string())
            .collect();
        let output_vec = model.run(input_tensor)?;

        let mut output_map = HashMap::new();
        for (name, tensor) in output_names.iter().zip(output_vec) {
            let extracted = tensor.try_extract()?;
            let view = extracted.view();
            let owned = view.to_owned();
            let dimensionality = owned.into_dimensionality::<IxDyn>()?;
            output_map.insert(name.to_string(), dimensionality);
        }

        let output_logit = output_map.remove("logits").unwrap();

        Ok(output_logit.into_dimensionality()?)
    }

    fn prepare_input_map(
        &self,
        input_ids: Array2<u32>,
        encoder_last_hidden_state: Array3<f32>,
        encoder_attention_mask: Option<Array2<u32>>,
    ) -> Result<HashMap<String, InputTensor>> {
        let encoder_attention_mask = if encoder_attention_mask.is_none() {
            Array::ones((
                encoder_last_hidden_state.shape()[0],
                encoder_last_hidden_state.shape()[1],
            ))
        } else {
            encoder_attention_mask.unwrap()
        };
        let mut input_map = HashMap::<String, InputTensor>::new();
        input_map.insert(
            "input_ids".to_string(),
            InputTensor::from_array(input_ids.into_dimensionality()?),
        );
        input_map.insert(
            "encoder_attention_mask".to_string(),
            InputTensor::from_array(encoder_attention_mask.into_dimensionality()?),
        );
        input_map.insert(
            "encoder_hidden_states".to_string(),
            InputTensor::from_array(encoder_last_hidden_state.into_dimensionality()?),
        );
        Ok(input_map)
    }

    pub fn get_token_type_support(&self) -> bool {
        self.token_type_support
    }
}

#[cfg(test)]
mod tests {
    use crate::hf_hub::hf_hub_download;
    use crate::seq2seq_encoder::Seq2SeqEncoderModel;

    use super::*;

    #[test]
    fn test_model() {
        let env = Environment::builder().build().unwrap();
        let arc = env.into_arc();
        let decoder_model = Seq2SeqDecoderModel::new_from_file(
            arc.clone(),
            hf_hub_download("optimum/t5-small", "decoder_model.onnx", None, None).unwrap(),
            Device::CPU,
            GraphOptimizationLevel::Level3,
        )
        .unwrap();
        let encoder_model = Seq2SeqEncoderModel::new_from_file(
            arc,
            hf_hub_download("optimum/t5-small", "encoder_model.onnx", None, None).unwrap(),
            Device::CPU,
            GraphOptimizationLevel::Level3,
        )
        .unwrap();
        let input = vec![0, 1, 23, 23, 23, 23, 23, 23, 23, 23];
        let ndarray_input = Array2::<u32>::from_shape_vec((1, 10), input.clone()).unwrap();

        let encoder_output = encoder_model
            .forward(ndarray_input.clone(), None, None)
            .unwrap();
        let decoder_output = decoder_model
            .forward(ndarray_input, encoder_output, None)
            .unwrap();

        println!("{:?}", decoder_output);
    }
}
