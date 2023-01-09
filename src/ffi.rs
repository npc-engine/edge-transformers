use std::ffi::CString;

use interoptopus::patterns::string::AsciiPointer;
use interoptopus::{
    ffi_service, ffi_service_ctor, ffi_service_method, ffi_type, pattern, Inventory,
    InventoryBuilder,
};
use onnxruntime::{environment::Environment, GraphOptimizationLevel, LoggingLevel};

use crate::common::Device;
use crate::error::Result;
use crate::ffi::error::FFIError;

pub mod conditional_generation;
pub mod conditional_generation_with_pkvs;
pub mod embedding;
pub mod error;
pub mod optimum_seq2seq_generation;
pub mod optimum_seq2seq_generation_with_pkvs;
pub mod seq2seq_generation;
pub mod sequence_classification;
pub mod token_classification;

// Environment wrapper

/// Holds text embedding with model specific threshold for cosine similarity.
#[ffi_type(opaque, name = "Environment")]
pub struct EnvContainer {
    pub env: Environment,
}

/// Holds text embedding with model specific threshold for cosine similarity.
#[ffi_service(error = "FFIError", prefix = "onnx_env_")]
impl EnvContainer {
    #[ffi_service_ctor]
    pub fn new() -> Result<Self> {
        let env = Environment::builder()
            .with_log_level(LoggingLevel::Error)
            .build()
            .unwrap();
        Ok(Self { env })
    }
}

#[ffi_type]
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GraphOptimizationLevelFFI {
    DisableAll = 0,
    Basic = 1,
    Extended = 2,
    All = 99,
}

impl From<GraphOptimizationLevelFFI> for GraphOptimizationLevel {
    fn from(level: GraphOptimizationLevelFFI) -> Self {
        match level {
            GraphOptimizationLevelFFI::DisableAll => GraphOptimizationLevel::DisableAll,
            GraphOptimizationLevelFFI::Basic => GraphOptimizationLevel::Basic,
            GraphOptimizationLevelFFI::Extended => GraphOptimizationLevel::Extended,
            GraphOptimizationLevelFFI::All => GraphOptimizationLevel::All,
        }
    }
}

impl From<GraphOptimizationLevel> for GraphOptimizationLevelFFI {
    fn from(level: GraphOptimizationLevel) -> Self {
        match level {
            GraphOptimizationLevel::DisableAll => GraphOptimizationLevelFFI::DisableAll,
            GraphOptimizationLevel::Basic => GraphOptimizationLevelFFI::Basic,
            GraphOptimizationLevel::Extended => GraphOptimizationLevelFFI::Extended,
            GraphOptimizationLevel::All => GraphOptimizationLevelFFI::All,
        }
    }
}

#[ffi_type]
#[repr(C)]
#[cfg(all(
    not(feature = "cuda"),
    not(feature = "tensorrt"),
    not(feature = "directml")
))]
pub enum DeviceFFI {
    CPU,
}

#[ffi_type]
#[repr(C)]
#[cfg(feature = "directml")]
pub enum DeviceFFI {
    CPU,
    DML,
}

#[ffi_type]
#[repr(C)]
#[cfg(any(feature = "cuda", feature = "tensorrt"))]
pub enum DeviceFFI {
    CPU,
    CUDA,
}

impl From<DeviceFFI> for Device {
    fn from(device: DeviceFFI) -> Self {
        match device {
            DeviceFFI::CPU => Device::CPU,
            #[cfg(feature = "directml")]
            DeviceFFI::DML => Device::DML,
            #[cfg(feature = "cuda")]
            DeviceFFI::CUDA => Device::CUDA,
        }
    }
}

#[repr(C)]
#[ffi_type(opaque, name = "StringBatch")]
pub struct StringBatch {
    batch: Vec<String>,
}

impl Default for StringBatch {
    fn default() -> Self {
        Self { batch: vec![] }
    }
}

#[ffi_service(error = "FFIError", prefix = "onnx_string_batch_")]
impl StringBatch {
    #[ffi_service_ctor]
    pub fn new() -> Result<Self> {
        Ok(Self { batch: vec![] })
    }

    #[ffi_service_method(on_panic = "ffi_error")]
    pub fn add(&mut self, add_string: AsciiPointer) -> Result<()> {
        let add_string = add_string.as_str()?.to_string();
        self.batch.push(add_string);
        Ok(())
    }

    #[ffi_service_method(on_panic = "return_default")]
    pub fn get(&self, id: u32) -> AsciiPointer {
        AsciiPointer::from_slice_with_nul(
            CString::new(self.batch[id as usize].clone())
                .unwrap()
                .into_bytes_with_nul()
                .as_slice(),
        )
        .expect("Failed to convert CString to AsciiPointer")
    }

    #[ffi_service_method(on_panic = "return_default")]
    pub fn length(&self) -> u32 {
        self.batch.len() as u32
    }

    #[ffi_service_method(on_panic = "ffi_error")]
    pub fn clear(&mut self) -> Result<()> {
        self.batch.clear();
        Ok(())
    }
}

#[ffi_type]
#[repr(C)]
pub struct UseAsciiStringPattern<'a> {
    pub ascii_string: AsciiPointer<'a>,
}

pub fn ffi_inventory() -> Inventory {
    {
        InventoryBuilder::new()
            // Environment
            .register(pattern!(crate::ffi::EnvContainer))
            .register(pattern!(crate::ffi::StringBatch))
            // ConditionalGenerationPipeline
            .register(pattern!(crate::ffi::conditional_generation::ConditionalGenerationPipelineFFI))
            .register(pattern!(crate::ffi::conditional_generation_with_pkvs::ConditionalGenerationPipelineWithPKVsFFI))
            // Embedding pipeline
            .register(pattern!(crate::ffi::embedding::EmbeddingPipelineFFI))
            // Sequence classification pipeline
            .register(pattern!(crate::ffi::sequence_classification::SequenceClassificationPipelineFFI))
            // Token classification pipeline
            .register(pattern!(crate::ffi::token_classification::TokenClassificationPipelineFFI))
            // Seq2Seq pipeline
            .register(pattern!(crate::ffi::optimum_seq2seq_generation::OptimumSeq2SeqPipelineFFI))
            .register(pattern!(crate::ffi::optimum_seq2seq_generation_with_pkvs::OptimumSeq2SeqPipelineWithPKVsFFI))
            .register(pattern!(crate::ffi::seq2seq_generation::Seq2SeqGenerationPipelineFFI))
            .inventory()
    }
}
