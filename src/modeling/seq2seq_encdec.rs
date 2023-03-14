use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use ort::environment::Environment;
use ndarray::{Array, Array2, Array3, IxDyn};
use ort::session::{Input, Output};
use ort::tensor::{FromArray, InputTensor};
use ort::{GraphOptimizationLevel, InMemorySession, Session, SessionBuilder};

use crate::common::Device;
use crate::common::{apply_device, match_to_inputs};
use crate::error::{Error, Result};
use crate::ORTSession;

/// Onnx inference session wrapper for the Seq2Seq generation models.
pub struct Seq2SeqGenerationModel<'a> {
    model_session: ORTSession<'a>,
    token_type_support: bool,
    decoder_token_type_support: bool,
}

impl<'a> Seq2SeqGenerationModel<'a> {
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
        let (token_type_support, decoder_token_type_support) =
            Self::validate_signature(&session.inputs, &session.outputs)?;
        Ok(Self {
            model_session: ORTSession::InMemory(session),
            token_type_support,
            decoder_token_type_support,
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
        let (token_type_support, decoder_token_type_support) =
            Self::validate_signature(&session.inputs, &session.outputs)?;
        Ok(Self {
            model_session: ORTSession::Owned(session),
            token_type_support,
            decoder_token_type_support,
        })
    }

    fn validate_signature(inputs: &Vec<Input>, outputs: &Vec<Output>) -> Result<(bool, bool)> {
        let token_type_support = inputs.iter().any(|input| input.name == "token_type_ids");
        let decoder_token_type_support = inputs
            .iter()
            .any(|input| input.name == "decoder_token_type_ids");
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
        if past_values.len() != 0 {
            return Err(Error::OnnxIncorrectInputs {
                message: "The model needs support past key values. Use ConditionalGenerationModelWithPKVs instead".to_string(),
                expected: vec!["input_ids".to_string(), "attention_mask".to_string()],
                actual: past_values,
            });
        }
        if present_values_mapped.len() != 0 {
            return Err(Error::OnnxIncorrectOutputs {
                message: "The model needs support past key values. Use ConditionalGenerationModelWithPKVs instead".to_string(),
                expected: vec!["logits".to_string()],
                actual: outputs
                    .iter()
                    .map(|output| output.name.to_string())
                    .filter(|output| output.contains("present"))
                    .collect(),
            });
        }

        if inputs.iter().all(|inp| inp.name != "input_ids")
            || inputs.iter().all(|inp| inp.name != "attention_mask")
            || inputs.iter().all(|inp| inp.name != "decoder_input_ids")
            || inputs
                .iter()
                .all(|inp| inp.name != "decoder_attention_mask")
        {
            return Err(Error::OnnxIncorrectInputs {
                message: "The model does not have the required inputs.".to_string(),
                actual: inputs.iter().map(|inp| inp.name.to_string()).collect(),
                expected: vec![
                    "input_ids".to_string(),
                    "attention_mask".to_string(),
                    "decoder_input_ids".to_string(),
                    "decoder_attention_mask".to_string(),
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
        Ok((token_type_support, decoder_token_type_support))
    }

    pub fn get_token_type_support(&self) -> bool {
        self.token_type_support
    }

    pub fn get_decoder_token_type_support(&self) -> bool {
        self.decoder_token_type_support
    }

    /// Does inference.
    /// Returns logits and the past key values.
    pub fn forward(
        &self,
        input_ids: Array2<u32>,
        attention_mask: Option<Array2<u32>>,
        decoder_input_ids: Array2<u32>,
        decoder_attention_mask: Option<Array2<u32>>,
        token_type_ids: Option<Array2<u32>>,
        decoder_token_type_ids: Option<Array2<u32>>,
    ) -> Result<Array3<f32>> {
        let input_map = self.prepare_input_map(
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            token_type_ids,
            decoder_token_type_ids,
        )?;
        let mut model = match self.model_session {
            ORTSession::Owned(ref model) => model,
            ORTSession::InMemory(ref model) => model,
        };
        let input_tensor = match_to_inputs(&model.inputs, input_map)?;
        let output_names: Vec<String> = model
            .outputs
            .iter()
            .map(|output| output.name.to_string())
            .collect();
        let output_vec = model.run(input_tensor)?;

        let mut output_map: HashMap<String, Array<f32, IxDyn>> = output_names
            .iter()
            .map(|name| name.to_string())
            .zip(output_vec.into_iter().map(|tensor| {
                tensor.try_extract().unwrap().view().to_owned()
            }))
            .collect();

        let output_logit = output_map.remove("logits").unwrap();

        Ok(output_logit.into_dimensionality()?)
    }

    fn prepare_input_map(
        &self,
        input_ids: Array2<u32>,
        attention_mask: Option<Array2<u32>>,
        decoder_input_ids: Array2<u32>,
        decoder_attention_mask: Option<Array2<u32>>,
        token_type_ids: Option<Array2<u32>>,
        decoder_token_type_ids: Option<Array2<u32>>,
    ) -> Result<HashMap<String, InputTensor>> {
        let attention_mask = if attention_mask.is_none() {
            Array::ones((input_ids.shape()[0], input_ids.shape()[1]))
        } else {
            attention_mask.unwrap()
        };
        let decoder_attention_mask = if decoder_attention_mask.is_none() {
            Array::ones((decoder_input_ids.shape()[0], decoder_input_ids.shape()[1]))
        } else {
            decoder_attention_mask.unwrap()
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
        let decoder_token_type_ids = if self.decoder_token_type_support {
            Some(if decoder_token_type_ids.is_none() {
                Array::zeros((decoder_input_ids.shape()[0], decoder_input_ids.shape()[1]))
            } else {
                decoder_token_type_ids.unwrap()
            })
        } else {
            None
        };
        let mut input_map = HashMap::<String, InputTensor>::new();
        if let Some(token_types_array) = token_type_ids {
            input_map.insert(
                "token_type_ids".to_string(),
                InputTensor::from_array(token_types_array.into_dimensionality()?),
            );
        }
        if let Some(decoder_token_types_array) = decoder_token_type_ids {
            input_map.insert(
                "decoder_token_type_ids".to_string(),
                InputTensor::from_array(decoder_token_types_array.into_dimensionality()?),
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
        input_map.insert(
            "decoder_input_ids".to_string(),
            InputTensor::from_array(decoder_input_ids.into_dimensionality()?),
        );
        input_map.insert(
            "decoder_attention_mask".to_string(),
            InputTensor::from_array(decoder_attention_mask.into_dimensionality()?),
        );
        Ok(input_map)
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Read;

    use super::*;

    /// seq2seq from transformers, can't test it with from_pretrained because it's not of optimum format
    #[ignore]
    #[test]
    fn test_model() -> Result<()> {
        let mut file = File::open("resources/seq2seq/model.onnx")?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let env = Environment::builder().build()?;
        let model = Seq2SeqGenerationModel::new_from_memory(
            env.into_arc(),
            buffer.as_slice(),
            Device::CPU,
            GraphOptimizationLevel::Level3,
        )?;
        let input = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let decoder_input = vec![0, 0, 0, 0, 0];
        let ndarray_input = Array2::<u32>::from_shape_vec((1, 10), input.clone())?;
        let ndarray_decoder_input = Array2::<u32>::from_shape_vec((1, 5), decoder_input.clone())?;

        let output = model.forward(ndarray_input, None, ndarray_decoder_input, None, None, None)?;
        assert!(!output.iter().any(|x| x.is_nan()));
        Ok(())
    }
}
