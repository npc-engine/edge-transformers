use std::borrow::Borrow;
use std::cell::{Ref, RefCell};
use std::ffi::{CStr, CString};
use std::path::Path;

use interoptopus::patterns::slice::FFISlice;
use interoptopus::patterns::string::AsciiPointer;
use interoptopus::{
    ffi_function, ffi_service, ffi_service_ctor, ffi_service_method, ffi_type, function, pattern,
    Inventory, InventoryBuilder,
};
use onnxruntime::ndarray::AssignElem;
use onnxruntime::{environment::Environment, LoggingLevel};
use thread_local::ThreadLocal;

use crate::common::Device;
use crate::error::Error;
use crate::error::Result;
use crate::ffi::error::FFIError;
use crate::ffi::{DeviceFFI, EnvContainer, GraphOptimizationLevelFFI, StringBatch};
use crate::sampling::{ArgmaxSampler, RandomSampler, TopKSampler};
use crate::{
    ClassPrediction, ConditionalGenerationPipeline, Embedding, EmbeddingPipeline, PoolingStrategy,
    Prediction, SequenceClassificationPipeline, TaggedString, TokenClassPrediction,
    TokenClassificationPipeline,
};

#[repr(C)]
#[ffi_type]
pub enum PoolingStrategyFFI {
    Mean,
    Max,
    First,
}

impl From<PoolingStrategyFFI> for PoolingStrategy {
    fn from(pooling_strategy: PoolingStrategyFFI) -> Self {
        match pooling_strategy {
            PoolingStrategyFFI::Mean => PoolingStrategy::Mean,
            PoolingStrategyFFI::Max => PoolingStrategy::Max,
            PoolingStrategyFFI::First => PoolingStrategy::First,
        }
    }
}

#[repr(C)]
#[ffi_type]
pub struct EmbeddingFFI<'a> {
    pub embedding: FFISlice<'a, f32>,
}

impl Default for EmbeddingFFI<'_> {
    fn default() -> Self {
        Self {
            embedding: FFISlice::default(),
        }
    }
}

impl<'a> From<&'a Embedding> for EmbeddingFFI<'a> {
    fn from(embedding: &'a Embedding) -> Self {
        Self {
            embedding: FFISlice::from_slice(
                embedding
                    .embedding
                    .as_slice()
                    .expect("Embedding is not contiguous"),
            ),
        }
    }
}

#[ffi_type(opaque, name = "EmbeddingPipeline")]
pub struct EmbeddingPipelineFFI<'a> {
    pub model: EmbeddingPipeline<'a>,
    output_buf: Vec<Embedding>,
    vec_output_buf: Vec<EmbeddingFFI<'a>>,
}

#[ffi_service(error = "FFIError", prefix = "onnx_emb_")]
impl<'a> EmbeddingPipelineFFI<'a> {
    #[ffi_service_ctor]
    pub fn from_pretrained(
        env: &'a EnvContainer,
        model_id: AsciiPointer<'a>,
        pooling_strategy: PoolingStrategyFFI,
        device: DeviceFFI,
        optimization: GraphOptimizationLevelFFI,
    ) -> Result<Self> {
        let model = EmbeddingPipeline::from_pretrained(
            env.env.borrow(),
            model_id.as_str().unwrap().to_string(),
            pooling_strategy.into(),
            device.into(),
            optimization.into(),
        )?;
        Ok(Self {
            model,
            output_buf: Vec::new(),
            vec_output_buf: Vec::new(),
        })
    }

    #[ffi_service_ctor]
    pub fn create_from_memory(
        env: &'a EnvContainer,
        model: FFISlice<u8>,
        tokenizer_config: AsciiPointer<'a>,
        special_tokens_map: AsciiPointer<'a>,
        pooling_strategy: PoolingStrategyFFI,
        device: DeviceFFI,
        optimization: GraphOptimizationLevelFFI,
    ) -> Result<Self> {
        let model = EmbeddingPipeline::<'a>::new_from_memory(
            &env.env,
            model.as_slice(),
            tokenizer_config.as_str().unwrap().to_string(),
            special_tokens_map.as_str().unwrap().to_string(),
            pooling_strategy.into(),
            device.into(),
            optimization.into(),
        )?;
        Ok(Self {
            model,
            output_buf: Vec::new(),
            vec_output_buf: Vec::new(),
        })
    }

    #[ffi_service_ctor]
    pub fn create_from_files(
        env: &'a EnvContainer,
        model_path: AsciiPointer<'a>,
        tokenizer_config_path: AsciiPointer<'a>,
        special_tokens_map_path: AsciiPointer<'a>,
        pooling_strategy: PoolingStrategyFFI,
        device: DeviceFFI,
        optimization: GraphOptimizationLevelFFI,
    ) -> Result<Self> {
        let model = EmbeddingPipeline::new_from_files(
            &env.env,
            Path::new(model_path.as_str().unwrap()).to_path_buf(),
            Path::new(tokenizer_config_path.as_str().unwrap()).to_path_buf(),
            Path::new(special_tokens_map_path.as_str().unwrap()).to_path_buf(),
            pooling_strategy.into(),
            device.into(),
            optimization.into(),
        )?;
        Ok(Self {
            model,
            output_buf: Vec::new(),
            vec_output_buf: Vec::new(),
        })
    }

    #[ffi_service_method(on_panic = "return_default")]
    pub fn embed(s: &'a mut EmbeddingPipelineFFI, input: AsciiPointer<'a>) -> EmbeddingFFI<'a> {
        let output = s.model.embed(input.as_str().unwrap()).unwrap();

        s.output_buf = vec![output];
        EmbeddingFFI::from(&s.output_buf[0])
    }

    #[ffi_service_method(on_panic = "return_default")]
    pub fn embed_batch(
        s: &'a mut EmbeddingPipelineFFI<'a>,
        input: StringBatch,
    ) -> FFISlice<'a, EmbeddingFFI<'a>> {
        let output = s.model.embed_batch(input.batch).unwrap();
        s.output_buf = output;
        s.vec_output_buf = s.output_buf.iter().map(|x| EmbeddingFFI::from(x)).collect();
        FFISlice::from_slice(&s.vec_output_buf)
    }
}
