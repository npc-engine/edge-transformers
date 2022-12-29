mod error;
mod sampling;
pub mod ffi;

use error::*;
use sampling::*;
use ffi::Device;

use std::path::Path;
use tokenizers::tokenizer::Tokenizer;

use onnxruntime::{s, Array, ArrayD, Axis};
use onnxruntime::tensor::{FromArray, InputTensor, OrtOwnedTensor};
use onnxruntime::{environment::Environment, session::Session, LoggingLevel};
use onnxruntime::{ndarray, GraphOptimizationLevel};
use std::convert::TryInto;
use tokenizers::{Encoding, PaddingDirection, PaddingParams, PaddingStrategy};

use crate::sampling::{sample, Sampler};
use onnxruntime::session::{Input, Output};
use std::cell::{RefCell, RefMut};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::str::FromStr;
use onnxruntime::ndarray::{Array, Array2, ArrayD, Ix2, IxDyn, s};


/// Wraps Huggingface ONNX exported transformer and tokenizer
/// into simple text to text generative interface.
pub struct ConditionalGenerationPipeline<'a> {
    tokenizer: Tokenizer,
    model_session: RefCell<Session<'a>>,
    token_type_support: bool,
    past_key_values: Vec<String>,
    eos_token_id: u32,
}


impl<'a> ConditionalGenerationPipeline<'a> {
    /// Creates a GenerativeModel from a path to a model folder.
    pub fn new_from_memory(
        environment: &'a Environment,
        model: &[u8],
        tokenizer_config: String,
        special_tokens_map: String,
        device: Device,
    ) -> Result<Self> {
        let mut session_builder = environment.new_session_builder()?;
        match device {
            Device::CPU => {
                session_builder = session_builder.use_cpu(1)?;
            }
            Device::DML => {
                session_builder = session_builder.use_dml()?;
            }
        }
        let session = session_builder
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_model_from_memory(model)?;

        let (token_type_support, past_key_values) =
            Self::validate_signature(&session.inputs, &session.outputs)?;

        let tok = Tokenizer::from_str(tokenizer_config.as_ref())?;
        /// Read json special tokens map
        let special_tokens_map: HashMap<String, HashMap<String, String>> = serde_json::from_str(special_tokens_map.as_ref())?;
        let eos_token = special_tokens_map.get("eos_token").unwrap().get("content").unwrap();
        let eos_token_id = tok.token_to_id(eos_token).unwrap();
        Ok(ConditionalGenerationPipeline {
            tokenizer: tok,
            model_session: RefCell::new(session),
            token_type_support,
            past_key_values,
            eos_token_id,
        })
    }

    pub fn new(
        environment: &'a Environment,
        model: &'a Path,
        tokenizer_config: &'a Path,
        special_tokens_map: &'a Path,
        device: Device,
    ) -> Result<Self> {
        Self::new_from_memory(
            environment,
            &fs::read(model)?,
            fs::read_to_string(tokenizer_config)?,
            fs::read_to_string(special_tokens_map)?,
            device,
        )
    }

    /// Generates text based on formatted prompt.
    pub fn generate<'sampler>(
        &self,
        prompt: &str,
        max_length: i32,
        sampler: &'sampler dyn Sampler,
    ) -> Result<String> {
        let tokens: Vec<i64> = self
            .tokenizer
            .encode(prompt, true)?
            .get_ids()
            .to_vec()
            .into_iter()
            .map(|u| i64::from(u))
            .collect();

        let ids = if self.past_key_values.is_empty() {
            self.generate_without_past(tokens, max_length, sampler)?
        } else {
            self.generate_with_past(tokens, max_length, sampler)?
        };
        Ok(self.tokenizer.decode(ids, true)?)
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

        if inputs.iter().all(|inp| inp.name != "input_ids")
            || inputs.iter().all(|inp| inp.name != "attention_mask")
        {
            return Err(Error {
                message: "Incorrect model input signature, only \
                    ('input_ids', 'attention_mask', ['token_type_ids'] [past_key_values*]) \
                    where optional arguments are in [] ."
                    .to_string(),
            });
        }
        if outputs.iter().all(|output| output.name != "logits")
            || present_values_mapped
                .iter()
                .enumerate()
                .any(|(id, value)| past_values[id] != *value)
        {
            return Err(Error {
                message: "Incorrect model output signature, only \
                    ('logits', [present*]) where optional arguments are in [] \
                    . `present*` outputs correspond to past_key_values* inputs."
                    .to_string(),
            });
        }

        Ok((token_type_support, past_values))
    }

    fn generate_with_past(
        &self,
        tokens: Vec<i64>,
        max_length: i32,
        sampler: &dyn Sampler,
    ) -> Result<Vec<u32>> {
        let original_tok_length = tokens.len();
        let mut output_tokens: Vec<u32> = Default::default();

        let past_key_values = self.initialize_past();
        let mask = Array::<f32, Ix2>::ones((1, original_tok_length));
        let token_types = if self.token_type_support {
            None
        } else {
            Some(Array::<f32, Ix2>::zeros((
                1,
                original_tok_length,
            )))
        };
        let mut model = self.model_session.borrow_mut();
        let output_names: Vec<String> = model
            .outputs
            .iter()
            .map(|output| (&output.name).replace("present", "past_key_values"))
            .collect();

        let mut input_tensor =
            Self::match_to_inputs(model, tokens.clone(), mask, token_types, past_key_values)?;

        for i in 1..max_length {
            model = self.model_session.borrow_mut();
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
                .map(|(output_name, tensor)| {
                    (output_name.replace("present", "past_key_values"), tensor)
                })
                .collect();
            let mask = Array::<f32, Ix2>::ones((
                1 as usize,
                original_tok_length + i as usize,
            ));
            let token_types = if self.token_type_support {
                None
            } else {
                Some(Array::<f32, Ix2>::zeros((
                    1 as usize,
                    original_tok_length + i as usize,
                )))
            };

            let max_id = sampler.sample(output_logit.slice(s![.., -1, ..]));
            output_tokens.push(max_id as u32);
            input_tensor = Self::match_to_inputs(
                model,
                vec![max_id as i64],
                mask,
                token_types,
                past_key_values,
            )?;
            if max_id == self.eos_token_id as usize {
                break;
            }
        }
        Ok(output_tokens)
    }

    fn initialize_past(&self) -> HashMap<String, Array<f32, IxDyn>> {
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
                                1
                            } else {
                                0
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

    fn match_to_inputs(
        model: RefMut<Session>,
        tokens: Vec<i64>,
        mask: Array2<f32>,
        token_types: Option<Array2<f32>>,
        pkvs: HashMap<String, Array<f32, IxDyn>>,
    ) -> Result<Vec<InputTensor<IxDyn>>> {
        let inputs = &model.inputs;
        let mut inputs_array_vector: Vec<InputTensor<IxDyn>> = Default::default();

        let mut input_map = HashMap::<String, InputTensor<IxDyn>>::new();
        for (past_key, past_value) in pkvs {
            input_map.insert(past_key, InputTensor::from_array(past_value));
        }
        if let Some(token_types_array) = token_types {
            input_map.insert(
                "token_type_ids".to_string(),
                InputTensor::from_array(token_types_array.into_dimensionality()?),
            );
        }
        input_map.insert(
            "input_ids".to_string(),
            InputTensor::from_array(
                Array::<i64, Ix2>::from_shape_vec((1, tokens.len()), tokens)
                    .unwrap()
                    .into_dimensionality()?,
            ),
        );
        input_map.insert(
            "attention_mask".to_string(),
            InputTensor::from_array(mask.into_dimensionality()?),
        );

        for input in inputs {
            inputs_array_vector.push(input_map.remove(&input.name).expect(&*format!(
                "Input  not found {:?} in {:?}",
                &input.name, input_map
            )));
        }
        Ok(inputs_array_vector)
    }

    fn generate_without_past(
        &self,
        mut tokens: Vec<i64>,
        max_length: i32,
        sampler: &dyn Sampler,
    ) -> Result<Vec<u32>> {
        let original_tok_length = tokens.len();
        let mut model = self.model_session.borrow_mut();

        for _ in 0..max_length {
            let tokens_array =
                InputTensor::from_array(Array::<i64, Ix2>::from_shape_vec(
                    (1, tokens.len()),
                    tokens.clone(),
                )?);
            let mask = InputTensor::from_array(Array::<i64, Ix2>::ones((
                1,
                tokens.len(),
            )));
            let mut input_tensor = vec![tokens_array, mask];
            if self.token_type_support {
                input_tensor.push(InputTensor::from_array(
                    Array::<i64, Ix2>::ones((1, tokens.len())),
                ));
            }

            let outputs: OrtOwnedTensor<f32, _> = model.run(input_tensor)?.pop().unwrap();
            let outputs =
                ArrayD::from_shape_vec(outputs.raw_dim(), outputs.iter().map(|x| *x).collect())?;
            let max_id = sampler.sample(outputs.slice(s![.., -1, ..]));
            tokens.push(max_id as i64);
            if max_id == self.eos_token_id as usize {
                break;
            }
        }
        Ok(tokens[original_tok_length..]
            .iter()
            .map(|x| *x as u32)
            .collect())
    }
}


#[cfg(test)]
mod tests {
    use crate::error::Result;
    use crate::Device;
    use crate::ConditionalGenerationPipeline;
    use onnxruntime::environment::Environment;
    use onnxruntime::LoggingLevel;
    use std::path::Path;
    use crate::sampling::TopKSampler;

    #[test]
    fn test_gen_model() -> Result<()> {
        let env = Environment::builder()
            .with_log_level(LoggingLevel::Error)
            .build()
            .unwrap();
        let model_path = Path::new("resources/generative_test/model.onnx");
        let config_path = Path::new("resources/generative_test/tokenizer.json");
        let special_tokens_path = Path::new("resources/generative_test/special_tokens_map.json");
        let pipeline = ConditionalGenerationPipeline::new(&env, model_path, config_path, special_tokens_path, Device::DML)?;
        let sampler = TopKSampler::new(5, 1.0);
        let output = pipeline.generate("Hello world", 10, &sampler)?;
        println!("{:?}", output);
        Ok(())
    }
}
