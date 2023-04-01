use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use ndarray::{Array2, ArrayD, Axis, IxDyn};
use ort::environment::Environment;
use ort::tensor::{FromArray, InputTensor};
use ort::{GraphOptimizationLevel, SessionBuilder};

use crate::common::Device;
use crate::common::{apply_device, match_to_inputs};
use crate::error::Result;
use crate::{try_extract_to_f32, ORTSession};

/// Onnx inference session wrapper for the conditional generation models.
pub struct ClassificationModel<'a> {
    model_session: ORTSession<'a>,
    token_type_support: bool,
    num_labels: usize,
    is_tok_classification: bool,
}

impl<'a> ClassificationModel<'a> {
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
        let num_dims = session.outputs[0].dimensions.len();
        let num_labels = session.outputs[0].dimensions[num_dims - 1].unwrap() as usize;
        let is_tok_classification = num_dims == 3;
        Ok(Self {
            model_session: ORTSession::InMemory(session),
            token_type_support,
            num_labels,
            is_tok_classification,
        })
    }

    pub fn new_from_file<'path>(
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
        let token_type_support = session.inputs.len() == 3
            && session
                .inputs
                .iter()
                .filter(|i| i.name == "token_type_ids")
                .count()
                > 0;
        let num_dims = session.outputs[0].dimensions.len();
        let num_labels = session.outputs[0].dimensions[num_dims - 1].unwrap() as usize;
        let is_tok_classification = num_dims == 3;
        Ok(Self {
            model_session: ORTSession::Owned(session),
            token_type_support,
            num_labels,
            is_tok_classification,
        })
    }

    /// Does inference
    ///
    /// Returns a vector of Embedding class that contains the embedding.
    pub fn forward(
        &self,
        input_ids: Array2<u32>,
        attention_mask: Array2<u32>,
        token_type_ids: Option<Array2<u32>>,
    ) -> Result<ArrayD<f32>> {
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
            let extracted = try_extract_to_f32(tensor)?;
            let view = extracted.view();
            let owned = view.to_owned();
            let dimensionality = owned.into_dimensionality::<IxDyn>()?;
            output_map.insert(name.to_string(), dimensionality);
        }
        let logits = output_map.remove("logits").unwrap();
        let exps = logits.mapv(|x: f32| x.exp());
        let sum = exps
            .sum_axis(Axis(logits.ndim() - 1))
            .insert_axis(Axis(logits.ndim() - 1));

        let softmax = exps / sum;
        Ok(softmax.into_dimensionality()?)
    }

    fn prepare_input_map(
        &self,
        input_ids: Array2<u32>,
        attention_mask: Array2<u32>,
        token_type_ids: Option<Array2<u32>>,
    ) -> Result<HashMap<String, InputTensor>> {
        let mut input_map = HashMap::<String, InputTensor>::new();

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

    pub fn get_token_type_support(&self) -> bool {
        self.token_type_support
    }

    pub fn get_num_labels(&self) -> usize {
        self.num_labels
    }

    pub fn is_token_classification(&self) -> bool {
        self.is_tok_classification
    }
}

#[cfg(test)]
mod tests {
    use crate::hf_hub::hf_hub_download;

    use super::*;

    #[test]
    fn test_seq_classify() {
        let env = Environment::builder().build().unwrap();
        let bert = ClassificationModel::new_from_file(
            env.into_arc(),
            hf_hub_download(
                "npc-engine/deberta-v3-small-finetuned-hate_speech18",
                "model.onnx",
                None,
                None,
            )
            .unwrap(),
            Device::CPU,
            GraphOptimizationLevel::Disable,
        )
        .unwrap();
        let input_ids1 =
            Array2::from_shape_vec((1, 8), vec![101, 2000, 1037, 1037, 1037, 1037, 1037, 102])
                .unwrap();
        let attention_mask = Array2::from_shape_vec((1, 8), vec![1, 1, 1, 1, 1, 1, 1, 1]).unwrap();
        let scores = bert.forward(input_ids1, attention_mask, None).unwrap();
        let allzeros = (scores.sum_axis(Axis(1)) - 1.0).mapv(|x| x.abs());
        assert_eq!(scores.len(), 4);
        assert!(allzeros.fold(true, |all_true, x| all_true && (*x < 1e-6)));
    }

    #[test]
    fn test_tok_classify() {
        let env = Environment::builder().build().unwrap();
        let bert = ClassificationModel::new_from_file(
            env.into_arc(),
            hf_hub_download("optimum/bert-base-NER", "model.onnx", None, None).unwrap(),
            Device::CPU,
            GraphOptimizationLevel::Disable,
        )
        .unwrap();
        let input_ids1 =
            Array2::from_shape_vec((1, 8), vec![101, 2000, 1037, 1037, 1037, 1037, 1037, 102])
                .unwrap();
        let attention_mask = Array2::from_shape_vec((1, 8), vec![1, 1, 1, 1, 1, 1, 1, 1]).unwrap();
        let scores = bert.forward(input_ids1, attention_mask, None).unwrap();
        let allzeros = (scores.sum_axis(Axis(2)) - 1.0).mapv(|x| x.abs());
        assert!(allzeros.fold(true, |all_true, x| all_true && (*x < 1e-6)));
    }
}
