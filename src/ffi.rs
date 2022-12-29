use std::ffi::{CStr, CString};
use onnxruntime::{environment::Environment, LoggingLevel};
use interoptopus::{
    ffi_function, ffi_service, ffi_service_ctor, ffi_service_method, ffi_type, function, pattern,
    Inventory, InventoryBuilder,
};
use interoptopus::patterns::slice::FFISlice;
use interoptopus::patterns::string::AsciiPointer;
use crate::ConditionalGenerationPipeline;
use crate::error::Error;
use crate::error::Result;
use crate::sampling::{ArgmaxSampler, RandomSampler, TopKSampler};

/// Device enum to specify the device to run the model on
#[ffi_type]
#[repr(C)]
#[derive(Eq, PartialEq, Hash, Debug)]
pub enum Device {
    CPU,
    DML,
}

// Environment wrapper

/// Holds text embedding with model specific threshold for cosine similarity.
#[ffi_type(opaque)]
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

// Pipeline wrappers

pub struct ResultString<'a> {
    pub result: String,
    pub result_raw: AsciiPointer<'a>
}


#[ffi_type(opaque, name = "ConditionalGenerationPipeline")]
pub struct ConditionalGenerationPipelineFFI<'a> {
    pub model: ConditionalGenerationPipeline<'a>,
}

/// TODO: Add return class with access to result string and a `free()` method.
#[ffi_service(error = "FFIError", prefix = "onnx_cond_gen_")]
impl<'a> ConditionalGenerationPipelineFFI<'a> {
    #[ffi_service_ctor]
    pub fn new(
        env: &'a EnvContainer,
        model: FFISlice<u8>,
        tokenizer_config: AsciiPointer<'a>,
        special_tokens_map: AsciiPointer<'a>,
        device: Device,
    ) -> Result<Self> {
        let model = ConditionalGenerationPipeline::new_from_memory(
            &env.env,
            model.as_slice(),
            tokenizer_config.as_str().unwrap().to_string(),
            special_tokens_map.as_str().unwrap().to_string(),
            device,
        )?;
        Ok(Self { model })
    }

    #[ffi_service_method(on_panic = "return_default")]
    pub fn generate_topk_sampling(&self, input: AsciiPointer<'a>, max_length: i32, topk: i32, temperature: f32) -> AsciiPointer<'a> {
        let sampler = TopKSampler::new(topk as usize, temperature);
        let output = self.model.generate(input.as_str().unwrap(), max_length, &sampler).unwrap();
        AsciiPointer::from_slice_with_nul(output.as_bytes()).unwrap()
    }

    #[ffi_service_method(on_panic = "return_default")]
    pub fn generate_random_sampling(&self, input: AsciiPointer<'a>, max_length: i32, temperature: f32) -> AsciiPointer<'a> {
        let sampler = RandomSampler::new(temperature);
        let output = self.model.generate(input.as_str().unwrap(), max_length, &sampler).unwrap();
        AsciiPointer::from_slice_with_nul(output.as_bytes()).unwrap()
    }

    #[ffi_service_method(on_panic = "return_default")]
    pub fn generate_argmax(&self, input: AsciiPointer<'a>, max_length: i32) -> AsciiPointer<'a> {
        let sampler = ArgmaxSampler::new();
        let output = self.model.generate(input.as_str().unwrap(), max_length, &sampler).unwrap();
        AsciiPointer::from_slice_with_nul(output.as_bytes()).unwrap()
    }



}


// Error handling wrapper

#[ffi_type(patterns(ffi_error))]
#[repr(C)]
pub enum FFIError {
    Ok = 0,
    Null = 100,
    Panic = 200,
    Fail = 300,
}

impl From<Error> for FFIError {
    fn from(_: Error) -> Self {
        Self::Fail
    }
}

impl Default for FFIError {
    fn default() -> Self {
        Self::Ok
    }
}

impl interoptopus::patterns::result::FFIError for FFIError {
    const SUCCESS: Self = Self::Ok;
    const NULL: Self = Self::Null;
    const PANIC: Self = Self::Panic;
}

impl std::error::Error for Error {}

pub fn ffi_inventory() -> Inventory {
    {
        InventoryBuilder::new()
            .register(pattern!(crate::ffi::EnvContainer))
            .register(pattern!(crate::ffi::ConditionalGenerationPipelineFFI))
            .inventory()
    }
}

