use std::cell::{RefCell, RefMut};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use onnxruntime::environment::Environment;
use onnxruntime::ndarray::{s, Array, Array2, Array3, ArrayD, Axis, Ix2, IxDyn};
use onnxruntime::session::{Input, Output, Session};
use onnxruntime::tensor::{FromArray, InputTensor, OrtOwnedTensor};
use onnxruntime::{GraphOptimizationLevel, LoggingLevel};
use tokenizers::Tokenizer;

use crate::common::Device;
use crate::common::{apply_device, match_to_inputs};
use crate::error::{Error, Result};

/// Onnx inference session wrapper for the conditional generation models.
///
/// Validates inputs and outputs of the model and provides a convenient interface to the model.
pub struct ConditionalGenerationModelWithPKVs<'a> {
    model_session: RefCell<Session<'a>>,
    token_type_support: bool,
    past_key_values: Vec<String>,
}

impl<'a> ConditionalGenerationModelWithPKVs<'a> {
    pub fn new_from_memory(
        env: &'a Environment,
        model_bytes: &[u8],
        device: Device,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<Self> {
        let mut session_builder = env.new_session_builder()?;
        session_builder = apply_device(session_builder, device)?;
        let session = session_builder
            .with_optimization_level(optimization_level)?
            .with_model_from_memory(model_bytes)?;
        let (token_type_support, past_key_values) =
            Self::validate_signature(&session.inputs, &session.outputs)?;
        Ok(Self {
            model_session: RefCell::new(session),
            token_type_support,
            past_key_values,
        })
    }

    pub fn new_from_file<'path>(
        env: &'a Environment,
        model_path: PathBuf,
        device: Device,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<Self> {
        let mut session_builder = env.new_session_builder()?;
        session_builder = apply_device(session_builder, device)?;
        let session = session_builder
            .with_optimization_level(optimization_level)?
            .with_model_from_file(model_path)?;
        let (token_type_support, past_key_values) =
            Self::validate_signature(&session.inputs, &session.outputs)?;
        Ok(Self {
            model_session: RefCell::new(session),
            token_type_support,
            past_key_values,
        })
    }

    fn validate_signature(
        inputs: &Vec<Input>,
        outputs: &Vec<Output>,
    ) -> Result<(bool, Vec<String>)> {
        let token_type_support = inputs.iter().any(|input| input.name == "token_type_ids");
        let past_values: Vec<String> = inputs
            .iter()
            .filter(|input| input.name.contains("past_key_values"))
            .map(|input| input.name.to_string())
            .collect();
        let present_values_mapped: Vec<String> = outputs
            .iter()
            .filter(|output| output.name.contains("present"))
            .map(|output| (&(output.name).to_string()).replace("present", "past_key_values"))
            .collect();

        if past_values.len() == 0 {
            return Err(Error::OnnxIncorrectInputs {
                message: "The model does not support past key values".to_string(),
                expected: vec!["past_key_values.*".to_string()],
                actual: past_values,
            });
        }
        if present_values_mapped.len() == 0 {
            return Err(Error::OnnxIncorrectOutputs {
                message: "The model does not support past key values".to_string(),
                expected: vec!["present.*".to_string()],
                actual: outputs
                    .iter()
                    .map(|output| output.name.to_string())
                    .filter(|output| output.contains("present"))
                    .collect(),
            });
        }

        if inputs.iter().all(|inp| inp.name != "input_ids")
            || inputs.iter().all(|inp| inp.name != "attention_mask")
        {
            return Err(Error::OnnxIncorrectInputs {
                message: "The model does not have the required inputs.".to_string(),
                actual: inputs.iter().map(|inp| inp.name.to_string()).collect(),
                expected: vec![
                    "input_ids".to_string(),
                    "attention_mask".to_string(),
                    "past_key_values.*".to_string(),
                ],
            });
        }
        if outputs.iter().all(|output| output.name != "logits") {
            return Err(Error::OnnxIncorrectOutputs {
                message: "The model does not have a logits output.".to_string(),
                actual: outputs.iter().map(|inp| inp.name.to_string()).collect(),
                expected: vec!["logits".to_string(), "present.*".to_string()],
            });
        }
        if present_values_mapped
            .iter()
            .enumerate()
            .any(|(id, value)| past_values[id] != *value)
        {
            return Err(Error::OnnxInputOutputMismatch {
                input: past_values,
                output: outputs
                    .iter()
                    .map(|output| output.name.to_string())
                    .filter(|output| output.contains("present"))
                    .collect(),
            });
        }
        Ok((token_type_support, past_values))
    }

    /// Does inference.
    /// Returns logits and the past key values.
    pub fn forward(
        &self,
        input_ids: Array2<u32>,
        attention_mask: Option<Array2<u32>>,
        token_type_ids: Option<Array2<u32>>,
        past_key_values: Option<HashMap<String, ArrayD<f32>>>,
    ) -> Result<(Array3<f32>, HashMap<String, ArrayD<f32>>)> {
        let input_map =
            self.prepare_input_map(input_ids, attention_mask, token_type_ids, past_key_values)?;
        let mut input_tensor = match_to_inputs(&self.model_session.borrow().inputs, input_map)?;
        let mut model = self.model_session.borrow_mut();
        let output_names: Vec<String> = model
            .outputs
            .iter()
            .map(|output| (&output.name).replace("present", "past_key_values"))
            .collect();

        let output_vec = model.run(input_tensor)?;

        let mut output_map: HashMap<String, Array<f32, IxDyn>> = output_names
            .iter()
            .map(|name| name.to_string())
            .zip(output_vec.into_iter().map(|tensor| {
                Array::<f32, IxDyn>::from_shape_vec(
                    tensor.shape(),
                    tensor.iter().map(|x| *x).collect(),
                )
                .unwrap()
            }))
            .collect();

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
        attention_mask: Option<Array2<u32>>,
        token_type_ids: Option<Array2<u32>>,
        past_key_values: Option<HashMap<String, ArrayD<f32>>>,
    ) -> Result<HashMap<String, InputTensor<IxDyn>>> {
        let past_key_values = if past_key_values.is_none() {
            self.initialize_past(input_ids.shape()[0] as u32)
        } else {
            past_key_values.unwrap()
        };
        let attention_mask = if attention_mask.is_none() {
            Array::ones((input_ids.shape()[0], input_ids.shape()[1]))
        } else {
            attention_mask.unwrap()
        };
        let token_type_ids = if self.token_type_support {
            Some(if token_type_ids.is_none() {
                Array::zeros((input_ids.shape()[0], input_ids.shape()[1]))
            } else {
                token_type_ids.unwrap()
            })
        } else {
            None
        };
        let mut input_map = HashMap::<String, InputTensor<IxDyn>>::new();
        for (past_key, past_value) in past_key_values {
            input_map.insert(past_key, InputTensor::from_array(past_value));
        }
        if let Some(token_types_array) = token_type_ids {
            input_map.insert(
                "token_type_ids".to_string(),
                InputTensor::from_array(token_types_array.into_dimensionality()?),
            );
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

    fn initialize_past(&self, batch_size: u32) -> HashMap<String, Array<f32, IxDyn>> {
        let model = self.model_session.borrow_mut();
        model
            .outputs
            .iter()
            .map(|output| {
                let shape: Vec<usize> = output
                    .dimensions
                    .iter()
                    .enumerate()
                    .map(|(id, dim)| match dim {
                        None => {
                            if id == 0 {
                                batch_size as usize
                            } else {
                                0 as usize
                            }
                        }
                        Some(size) => *size as usize,
                    })
                    .collect();
                (
                    output.name.replace("present", "past_key_values"),
                    ArrayD::<f32>::from_shape_vec(shape, Default::default()).unwrap(),
                )
            })
            .collect()
    }

    pub fn get_token_type_support(&self) -> bool {
        self.token_type_support
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Read;

    use crate::hf_hub::hf_hub_download;

    use super::*;

    #[test]
    fn test_model() {
        let env = Environment::builder().build().unwrap();
        let model = ConditionalGenerationModelWithPKVs::new_from_file(
            &env,
            hf_hub_download("optimum/gpt2", "decoder_with_past_model.onnx", None, None).unwrap(),
            Device::CPU,
            GraphOptimizationLevel::All,
        )
        .unwrap();
        let input = vec![
            50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
        ];
        let ndarray_input = Array2::<u32>::from_shape_vec((1, 10), input.clone()).unwrap();

        let output = model.forward(ndarray_input, None, None, None).unwrap();
        assert_eq!(output.0.shape(), &[1, 10, 50257]);
    }
}
