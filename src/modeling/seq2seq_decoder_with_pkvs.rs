use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use ort::environment::Environment;
use ndarray::{Array, Array2, Array3, ArrayD, IxDyn};
use ort::session::{Input, Output, Session};
use ort::tensor::{FromArray, InputTensor};
use ort::{GraphOptimizationLevel, InMemorySession, SessionBuilder};

use crate::{clone, ORTSession};
use crate::common::Device;
use crate::common::{apply_device, match_to_inputs};
use crate::error::{Error, Result};

/// Onnx inference session wrapper for the conditional generation models.
///
/// Validates inputs and outputs of the model and provides a convenient interface to the model.
pub struct Seq2SeqDecoderModelWithPKVs<'a> {
    model_session: ORTSession<'a>,
    model_session_with_pkvs: ORTSession<'a>,
    token_type_support: bool,
}

impl<'a> Seq2SeqDecoderModelWithPKVs<'a> {
    pub fn new_from_memory(
        env: Arc<Environment>,
        model_bytes: &'a [u8],
        model_with_pkvs_bytes: &'a [u8],
        device: Device,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<Self> {
        let mut session_builder = SessionBuilder::new(&env)?;
        session_builder = apply_device(session_builder, device.clone())?;
        let session = session_builder
            .with_optimization_level(clone(&optimization_level))?
            .with_model_from_memory(model_bytes)?;

        let mut session_builder = SessionBuilder::new(&env)?;
        session_builder = apply_device(session_builder, device)?;
        let session_pkvs = session_builder
            .with_optimization_level(optimization_level)?
            .with_model_from_memory(model_with_pkvs_bytes)?;

        Self::validate_signature(&session.inputs, &session.outputs, false)?;
        let token_type_support =
            Self::validate_signature(&session_pkvs.inputs, &session_pkvs.outputs, true)?;
        Ok(Self {
            model_session: ORTSession::InMemory(session),
            model_session_with_pkvs: ORTSession::InMemory(session_pkvs),
            token_type_support,
        })
    }

    pub fn new_from_file(
        env: Arc<Environment>,
        model_path: PathBuf,
        model_with_pkvs_path: PathBuf,
        device: Device,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<Self> {
        let mut session_builder = SessionBuilder::new(&env.clone())?;
        session_builder = apply_device(session_builder, device.clone())?;
        let session = session_builder
            .with_optimization_level(clone(&optimization_level))?
            .with_model_from_file(model_path)?;

        let mut session_builder = SessionBuilder::new(&env)?;
        session_builder = apply_device(session_builder, device)?;
        let session_pkvs = session_builder
            .with_optimization_level(optimization_level)?
            .with_model_from_file(model_with_pkvs_path)?;

        Self::validate_signature(&session.inputs, &session.outputs, false)?;
        let token_type_support =
            Self::validate_signature(&session_pkvs.inputs, &session_pkvs.outputs, true)?;
        Ok(Self {
            model_session: ORTSession::Owned(session),
            model_session_with_pkvs: ORTSession::Owned(session_pkvs),
            token_type_support,
        })
    }

    fn validate_signature(
        inputs: &Vec<Input>,
        outputs: &Vec<Output>,
        contain_past: bool,
    ) -> Result<bool> {
        let token_type_support = inputs.iter().any(|input| input.name == "token_type_ids");
        let past_values: Vec<String> = inputs
            .iter()
            .filter(|input| input.name.contains("past_key_values"))
            .map(|input| input.name.to_string())
            .collect();
        if contain_past {
            if past_values.len() == 0 {
                return Err(Error::OnnxIncorrectInputs {
                    message: "The model does not support past key values. Use Seq2SeqDecoderModel instead".to_string(),
                    expected: vec!["input_ids".to_string(), "attention_mask".to_string(), "past_key_values.*".to_string()],
                    actual: inputs.iter().map(|inp| inp.name.to_string()).collect(),
                });
            }
        } else {
            if past_values.len() != 0 {
                return Err(Error::OnnxIncorrectInputs {
                    message: "You should provide decoder model with no past key value along with one that supports PKVs.".to_string(),
                    expected: vec!["input_ids".to_string(), "attention_mask".to_string(), "past_key_values.*".to_string()],
                    actual: inputs.iter().map(|inp| inp.name.to_string()).collect(),
                });
            }
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
    /// Returns logits and the past key values.
    pub fn forward(
        &self,
        input_ids: Array2<u32>,
        encoder_last_hidden_state: Array3<f32>,
        encoder_attention_mask: Option<Array2<u32>>,
        past_key_values: Option<HashMap<String, ArrayD<f32>>>,
    ) -> Result<(Array3<f32>, HashMap<String, ArrayD<f32>>)> {
        let (input_map, use_pkvs) = self.prepare_input_map(
            input_ids,
            encoder_last_hidden_state,
            encoder_attention_mask,
            past_key_values,
        )?;
        let mut session = if use_pkvs {
            match &self.model_session_with_pkvs {
                ORTSession::InMemory(session) => session,
                ORTSession::Owned(session) => session,
            }
        } else {
            match &self.model_session {
                ORTSession::InMemory(session) => session,
                ORTSession::Owned(session) => session,
            }
        };
        let input_tensor = match_to_inputs(&session.inputs, input_map)?;

        let output_names: Vec<String> = session
            .outputs
            .iter()
            .map(|output| (&output.name).replace("present", "past_key_values"))
            .collect();

        let output_vec = session.run(input_tensor)?;

        let mut output_map = HashMap::new();
        for (name, tensor) in output_names.iter().zip(output_vec) {
            let extracted = tensor.try_extract()?;
            let view = extracted.view();
            let owned = view.to_owned();
            let dimensionality = owned.into_dimensionality::<IxDyn>()?;
            output_map.insert(name.to_string(), dimensionality);
        }

        let output_logit = output_map.remove("logits").unwrap();
        let past_key_values = output_map
            .into_iter()
            .filter(|(key, _)| key.starts_with("present") || key.starts_with("past_key_values"))
            .map(|(output_name, tensor)| {
                (output_name.replace("present", "past_key_values"), tensor)
            })
            .collect();
        Ok((output_logit.into_dimensionality()?, past_key_values))
    }

    fn prepare_input_map(
        &self,
        input_ids: Array2<u32>,
        encoder_last_hidden_state: Array3<f32>,
        encoder_attention_mask: Option<Array2<u32>>,
        past_key_values: Option<HashMap<String, ArrayD<f32>>>,
    ) -> Result<(HashMap<String, InputTensor>, bool)> {
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
        let use_pkvs = past_key_values.is_some();
        if use_pkvs {
            for (past_key, past_value) in past_key_values.unwrap() {
                input_map.insert(past_key, InputTensor::from_array(past_value));
            }
        }
        Ok((input_map, use_pkvs))
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
        let env = Environment::builder().build().unwrap().into_arc();
        let decoder_model = Seq2SeqDecoderModelWithPKVs::new_from_file(
            env.clone(),
            hf_hub_download("optimum/t5-small", "decoder_model.onnx", None, None).unwrap(),
            hf_hub_download(
                "optimum/t5-small",
                "decoder_with_past_model.onnx",
                None,
                None,
            )
            .unwrap(),
            Device::CPU,
            GraphOptimizationLevel::Level3,
        )
        .unwrap();
        let encoder_model = Seq2SeqEncoderModel::new_from_file(
            env,
            hf_hub_download("optimum/t5-small", "encoder_model.onnx", None, None).unwrap(),
            Device::CPU,
            GraphOptimizationLevel::Level3,
        )
        .unwrap();
        let input = vec![0, 1, 23, 23, 23, 23, 23, 23, 23, 23];
        let ndarray_input = Array2::<u32>::from_shape_vec((1, 10), input.clone()).unwrap();
        let decoder_ids = Array2::<u32>::from_shape_vec((1, 1), vec![1]).unwrap();

        let encoder_output = encoder_model
            .forward(ndarray_input.clone(), None, None)
            .unwrap();
        let decoder_output = decoder_model
            .forward(decoder_ids, encoder_output, None, None)
            .unwrap();

        println!("{:?}", decoder_output);
    }
}
