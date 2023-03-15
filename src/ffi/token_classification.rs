use std::borrow::Borrow;
use std::ffi::CString;
use std::path::Path;
use std::rc::Rc;

use interoptopus::patterns::slice::FFISlice;
use interoptopus::patterns::string::AsciiPointer;
use interoptopus::{ffi_service, ffi_service_ctor, ffi_service_method, ffi_type};

use crate::error::Result;
use crate::ffi::{
    error::FFIError, DeviceFFI, EnvContainer, GraphOptimizationLevelFFI, StringBatch,
};
use crate::{ClassPrediction, TaggedString, TokenClassPrediction, TokenClassificationPipeline};

#[repr(C)]
#[ffi_type]
pub struct TaggedStringFFI<'a> {
    pub input_string: AsciiPointer<'a>,
    pub tags: FFISlice<'a, TokenClassPredictionFFI<'a>>,
}

impl<'a> TaggedStringFFI<'a> {
    pub fn from_tagged_string(
        tagged_string: &'a TaggedString,
        tags_ffi_buf: &'a Vec<TokenClassPredictionFFI<'a>>,
    ) -> Self {
        let input_string = AsciiPointer::from_slice_with_nul(
            CString::new(tagged_string.input_string.as_bytes())
                .unwrap()
                .as_bytes_with_nul(),
        )
        .expect("Failed to convert input string to AsciiPointer");
        let tags = FFISlice::from_slice(tags_ffi_buf.as_slice());
        Self { input_string, tags }
    }
}

impl Default for TaggedStringFFI<'_> {
    fn default() -> Self {
        Self {
            input_string: AsciiPointer::from_slice_with_nul(b"").unwrap(),
            tags: FFISlice::from_slice(&[]),
        }
    }
}

#[repr(C)]
#[ffi_type]
pub struct TokenClassPredictionFFI<'a> {
    pub best: ClassPredictionFFI<'a>,
    pub all: FFISlice<'a, ClassPredictionFFI<'a>>,
    pub start: u32,
    pub end: u32,
}

impl<'a> TokenClassPredictionFFI<'a> {
    pub fn from_token_class_prediction(
        token_class_prediction: &'a TokenClassPrediction,
        prediction_all_ffi: &'a Vec<ClassPredictionFFI<'a>>,
    ) -> Self {
        let best = ClassPredictionFFI::from(&token_class_prediction.best);
        let all = FFISlice::from_slice(prediction_all_ffi.as_slice());
        let start = token_class_prediction.start;
        let end = token_class_prediction.end;
        Self {
            best,
            all,
            start: start as u32,
            end: end as u32,
        }
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

#[ffi_type(opaque)]
pub struct TokenClassificationPipelineFFI<'a> {
    pub model: TokenClassificationPipeline<'a>,
    output_buf: Vec<TaggedString>,
    output_buf_ffi: Vec<TaggedStringFFI<'a>>,
    tags_buf_ffi: Vec<Vec<TokenClassPredictionFFI<'a>>>,
    prediction_all_buf_ffi: Vec<Vec<Vec<ClassPredictionFFI<'a>>>>,
}

#[ffi_service(error = "FFIError", prefix = "onnx_token_classification_")]
impl<'a> TokenClassificationPipelineFFI<'a> {
    #[ffi_service_ctor]
    pub fn from_pretrained(
        env: &'a EnvContainer,
        model_id: AsciiPointer<'a>,
        device: DeviceFFI,
        optimization: GraphOptimizationLevelFFI,
    ) -> Result<Self> {
        let model = TokenClassificationPipeline::from_pretrained(
            env.env.clone(),
            model_id.as_c_str().unwrap().to_string_lossy().to_string(),
            device.into(),
            optimization.into(),
        )?;
        Ok(Self {
            model,
            output_buf: Vec::new(),
            output_buf_ffi: Vec::new(),
            tags_buf_ffi: Vec::new(),
            prediction_all_buf_ffi: Vec::new(),
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
        let model = TokenClassificationPipeline::new_from_files(
            env.env.clone(),
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
            output_buf_ffi: Vec::new(),
            tags_buf_ffi: Vec::new(),
            prediction_all_buf_ffi: Vec::new(),
        })
    }

    #[ffi_service_method(on_panic = "return_default")]
    pub fn tag(
        s: &'a mut TokenClassificationPipelineFFI<'a>,
        input: AsciiPointer,
    ) -> TaggedStringFFI<'a> {
        let input_str = input.as_c_str().unwrap().to_string_lossy().to_string();
        let output = s.model.tag(&input_str).unwrap();
        s.output_buf = vec![output];
        s.prediction_all_buf_ffi = vec![s.output_buf[0]
            .tags
            .iter()
            .map(|x| x.all.iter().map(|x| ClassPredictionFFI::from(x)).collect())
            .collect()];
        s.tags_buf_ffi = vec![s.output_buf[0]
            .tags
            .iter()
            .zip(s.prediction_all_buf_ffi.last().unwrap().iter())
            .map(|(x, y)| TokenClassPredictionFFI::from_token_class_prediction(x, y))
            .collect()];
        TaggedStringFFI::from_tagged_string(&s.output_buf[0], s.tags_buf_ffi.last().unwrap())
    }

    #[ffi_service_method(on_panic = "return_default")]
    pub fn tag_batch(
        s: &'a mut TokenClassificationPipelineFFI<'a>,
        input: StringBatch,
    ) -> FFISlice<'a, TaggedStringFFI<'a>> {
        let output = s.model.tag_batch(input.batch).unwrap();
        s.output_buf = output;
        s.prediction_all_buf_ffi = s
            .output_buf
            .iter()
            .map(|x| {
                x.tags
                    .iter()
                    .map(|x| x.all.iter().map(|x| x.into()).collect())
                    .collect()
            })
            .collect();
        s.tags_buf_ffi = s
            .output_buf
            .iter()
            .zip(s.prediction_all_buf_ffi.iter())
            .map(|(x, y)| {
                x.tags
                    .iter()
                    .zip(y.iter())
                    .map(|(x, y)| TokenClassPredictionFFI::from_token_class_prediction(x, y))
                    .collect()
            })
            .collect();
        s.output_buf_ffi = s
            .output_buf
            .iter()
            .zip(s.tags_buf_ffi.iter())
            .map(|(x, y)| TaggedStringFFI::from_tagged_string(x, y))
            .collect();
        FFISlice::from_slice(s.output_buf_ffi.as_slice())
    }
}
