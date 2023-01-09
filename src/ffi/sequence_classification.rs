use std::borrow::Borrow;
use std::ffi::CString;
use std::path::Path;

use interoptopus::patterns::slice::FFISlice;
use interoptopus::patterns::string::AsciiPointer;
use interoptopus::{ffi_service, ffi_service_ctor, ffi_service_method, ffi_type};

use crate::error::Result;
use crate::ffi::{
    error::FFIError, DeviceFFI, EnvContainer, GraphOptimizationLevelFFI, StringBatch,
};
use crate::{ClassPrediction, Prediction, SequenceClassificationPipeline};

#[repr(C)]
#[ffi_type]
pub struct PredictionFFI<'a> {
    pub best: ClassPredictionFFI<'a>,
    pub all: FFISlice<'a, ClassPredictionFFI<'a>>,
}

impl Default for PredictionFFI<'_> {
    fn default() -> Self {
        PredictionFFI {
            best: ClassPredictionFFI {
                label: AsciiPointer::default(),
                score: 0.0,
            },
            all: FFISlice::default(),
        }
    }
}

impl<'a> PredictionFFI<'a> {
    pub fn from_prediction(
        prediction: &'a Prediction,
        prediction_all_ffi: &'a Vec<ClassPredictionFFI<'a>>,
    ) -> Self {
        let best = ClassPredictionFFI::from(&prediction.best);
        let all = FFISlice::from_slice(prediction_all_ffi.as_slice());
        Self { best, all }
    }
}

#[repr(C)]
#[ffi_type]
pub struct ClassPredictionFFI<'a> {
    pub label: AsciiPointer<'a>,
    pub score: f32,
}

impl<'a> From<&ClassPrediction> for ClassPredictionFFI<'a> {
    fn from(class_prediction: &ClassPrediction) -> Self {
        let v = Vec::from(
            CString::new(class_prediction.label.as_str())
                .unwrap()
                .to_bytes_with_nul(),
        );
        Self {
            label: AsciiPointer::from_slice_with_nul(v.as_ref())
                .expect("Failed to convert CString to AsciiPointer"),
            score: class_prediction.score,
        }
    }
}

#[ffi_type(opaque, name = "SequenceClassificationPipeline")]
pub struct SequenceClassificationPipelineFFI<'a> {
    pub model: SequenceClassificationPipeline<'a>,
    output_buf: Vec<Prediction>,
    output_buf_vec: Vec<PredictionFFI<'a>>,
    class_preds_buf_vec: Vec<Vec<ClassPredictionFFI<'a>>>,
}

#[ffi_service(error = "FFIError", prefix = "onnx_classification_")]
impl<'a> SequenceClassificationPipelineFFI<'a> {
    #[ffi_service_ctor]
    pub fn from_pretrained(
        env: &'a EnvContainer,
        model_id: AsciiPointer<'a>,
        device: DeviceFFI,
        optimization: GraphOptimizationLevelFFI,
    ) -> Result<Self> {
        let model = SequenceClassificationPipeline::from_pretrained(
            env.env.borrow(),
            model_id.as_str().unwrap().to_string(),
            device.into(),
            optimization.into(),
        )?;
        Ok(Self {
            model,
            output_buf: Vec::new(),
            output_buf_vec: Vec::new(),
            class_preds_buf_vec: Vec::new(),
        })
    }

    #[ffi_service_ctor]
    pub fn create_from_memory(
        env: &'a EnvContainer,
        model: FFISlice<u8>,
        tokenizer_config: AsciiPointer<'a>,
        special_tokens_map: AsciiPointer<'a>,
        device: DeviceFFI,
        optimization: GraphOptimizationLevelFFI,
    ) -> Result<Self> {
        let model = SequenceClassificationPipeline::new_from_memory(
            &env.env,
            model.as_slice(),
            tokenizer_config.as_str().unwrap().to_string(),
            special_tokens_map.as_str().unwrap().to_string(),
            device.into(),
            optimization.into(),
            None,
        )?;
        Ok(Self {
            model,
            output_buf: Vec::new(),
            output_buf_vec: Vec::new(),
            class_preds_buf_vec: Vec::new(),
        })
    }

    #[ffi_service_ctor]
    pub fn create_from_files(
        env: &'a EnvContainer,
        model_path: AsciiPointer<'a>,
        tokenizer_config_path: AsciiPointer<'a>,
        special_tokens_map_path: AsciiPointer<'a>,
        device: DeviceFFI,
        optimization: GraphOptimizationLevelFFI,
    ) -> Result<Self> {
        let model = SequenceClassificationPipeline::new_from_files(
            &env.env,
            Path::new(model_path.as_str().unwrap()).to_path_buf(),
            Path::new(tokenizer_config_path.as_str().unwrap()).to_path_buf(),
            Path::new(special_tokens_map_path.as_str().unwrap()).to_path_buf(),
            device.into(),
            optimization.into(),
            None,
        )?;
        Ok(Self {
            model,
            output_buf: Vec::new(),
            output_buf_vec: Vec::new(),
            class_preds_buf_vec: Vec::new(),
        })
    }

    #[ffi_service_method(on_panic = "return_default")]
    pub fn classify(
        s: &'a mut SequenceClassificationPipelineFFI,
        input: AsciiPointer,
    ) -> PredictionFFI<'a> {
        let output = s.model.classify(input.as_str().unwrap()).unwrap();
        s.output_buf = vec![output];
        s.class_preds_buf_vec.push(
            s.output_buf
                .last()
                .unwrap()
                .all
                .iter()
                .map(|x| ClassPredictionFFI::from(x))
                .collect(),
        );
        let output_ffi = PredictionFFI::from_prediction(
            &s.output_buf.last().unwrap(),
            s.class_preds_buf_vec.last().unwrap(),
        );
        output_ffi
    }

    #[ffi_service_method(on_panic = "return_default")]
    pub fn classify_batch(
        s: &'a mut SequenceClassificationPipelineFFI<'a>,
        input: StringBatch,
    ) -> FFISlice<'a, PredictionFFI<'a>> {
        let output = s.model.classify_batch(input.batch).unwrap();
        s.output_buf = output;
        s.class_preds_buf_vec = s
            .output_buf
            .iter()
            .map(|x| x.all.iter().map(|x| ClassPredictionFFI::from(x)).collect())
            .collect();
        s.output_buf_vec = s
            .output_buf
            .iter()
            .zip(s.class_preds_buf_vec.iter())
            .map(|(x, y)| PredictionFFI::from_prediction(x, y))
            .collect();
        let output_ffi = FFISlice::from_slice(s.output_buf_vec.as_slice());
        output_ffi
    }
}
