use std::ffi::CString;
use std::path::PathBuf;

use interoptopus::patterns::slice::FFISlice;
use interoptopus::patterns::string::AsciiPointer;
use interoptopus::{ffi_service, ffi_service_ctor, ffi_service_method, ffi_type};

use crate::error::Result;
use crate::ffi::UseAsciiStringPattern;
use crate::ffi::{
    error::FFIError, DeviceFFI, EnvContainer, GraphOptimizationLevelFFI, StringBatch,
};
use crate::sampling::{ArgmaxSampler, RandomSampler, TopKSampler};
use crate::ConditionalGenerationPipelineWithPKVs;

#[ffi_type(opaque)]
pub struct ConditionalGenerationPipelineWithPKVsFFI<'a> {
    pub model: ConditionalGenerationPipelineWithPKVs<'a>,
    pub output_buf: Vec<String>,
    pub output_buf_ffi: Vec<UseAsciiStringPattern<'a>>,
}

#[ffi_service(error = "FFIError", prefix = "onnx_cond_gen_pkvs_")]
impl<'a> ConditionalGenerationPipelineWithPKVsFFI<'a> {
    #[ffi_service_ctor]
    pub fn from_pretrained(
        env: &'a EnvContainer,
        model_id: AsciiPointer<'a>,
        device: DeviceFFI,
        optimization: GraphOptimizationLevelFFI,
    ) -> Result<Self> {
        let model = ConditionalGenerationPipelineWithPKVs::from_pretrained(
            env.env.clone(),
            model_id.as_c_str().unwrap().to_string_lossy().to_string(),
            device.into(),
            optimization.into(),
        )?;
        Ok(Self {
            model,
            output_buf: Vec::new(),
            output_buf_ffi: Vec::new(),
        })
    }

    #[ffi_service_ctor]
    pub fn create_from_paths(
        env: &'a EnvContainer,
        model: AsciiPointer<'a>,
        tokenizer_config: AsciiPointer<'a>,
        special_tokens_map: AsciiPointer<'a>,
        device: DeviceFFI,
        optimization: GraphOptimizationLevelFFI,
    ) -> Result<Self> {
        let model = ConditionalGenerationPipelineWithPKVs::new_from_files(
            env.env.clone(),
            PathBuf::from(model.as_c_str().unwrap().to_string_lossy().to_string()),
            PathBuf::from(
                tokenizer_config
                    .as_c_str()
                    .unwrap()
                    .to_string_lossy()
                    .to_string(),
            ),
            PathBuf::from(
                special_tokens_map
                    .as_c_str()
                    .unwrap()
                    .to_string_lossy()
                    .to_string(),
            ),
            device.into(),
            optimization.into(),
        )?;
        Ok(ConditionalGenerationPipelineWithPKVsFFI {
            model,
            output_buf: Vec::new(),
            output_buf_ffi: Vec::new(),
        })
    }

    #[ffi_service_method(on_panic = "return_default")]
    pub fn generate_topk_sampling(
        &mut self,
        input: AsciiPointer,
        max_length: i32,
        topk: i32,
        temperature: f32,
    ) -> AsciiPointer<'a> {
        let sampler = TopKSampler::new(topk as usize, temperature);
        let output = self
            .model
            .generate(
                &*input.as_c_str().unwrap().to_string_lossy(),
                max_length,
                &sampler,
            )
            .unwrap();
        AsciiPointer::from_slice_with_nul(CString::new(output).unwrap().to_bytes_with_nul())
            .expect("Failed to convert CString to AsciiPointer")
    }

    #[ffi_service_method(on_panic = "return_default")]
    pub fn generate_random_sampling(
        &mut self,
        input: AsciiPointer<'a>,
        max_length: i32,
        temperature: f32,
    ) -> AsciiPointer<'a> {
        let sampler = RandomSampler::new(temperature);
        let output = self
            .model
            .generate(
                &*input.as_c_str().unwrap().to_string_lossy(),
                max_length,
                &sampler,
            )
            .unwrap();
        AsciiPointer::from_slice_with_nul(CString::new(output).unwrap().to_bytes_with_nul())
            .expect("Failed to convert CString to AsciiPointer")
    }

    #[ffi_service_method(on_panic = "return_default")]
    pub fn generate_argmax(
        &mut self,
        input: AsciiPointer<'a>,
        max_length: i32,
    ) -> AsciiPointer<'a> {
        let sampler = ArgmaxSampler::new();
        let output = self
            .model
            .generate(
                &*input.as_c_str().unwrap().to_string_lossy(),
                max_length,
                &sampler,
            )
            .unwrap();
        AsciiPointer::from_slice_with_nul(CString::new(output).unwrap().to_bytes_with_nul())
            .expect("Failed to convert CString to AsciiPointer")
    }

    #[ffi_service_method(on_panic = "return_default")]
    pub fn generate_topk_sampling_batch(
        s: &'a mut ConditionalGenerationPipelineWithPKVsFFI,
        input: StringBatch,
        max_length: i32,
        topk: i32,
        temperature: f32,
    ) -> FFISlice<'a, UseAsciiStringPattern<'a>> {
        let sampler = TopKSampler::new(topk as usize, temperature);
        s.output_buf = s
            .model
            .generate_batch(input.batch, max_length, &sampler)
            .unwrap();
        s.output_buf_ffi = s
            .output_buf
            .iter()
            .map(|s| {
                AsciiPointer::from_slice_with_nul(
                    CString::new(s.as_str()).unwrap().to_bytes_with_nul(),
                )
                .expect("Failed to convert CString to AsciiPointer")
            })
            .map(|s| UseAsciiStringPattern { ascii_string: s })
            .collect();
        FFISlice::from_slice(s.output_buf_ffi.as_slice())
    }

    #[ffi_service_method(on_panic = "return_default")]
    pub fn generate_random_sampling_batch(
        s: &'a mut ConditionalGenerationPipelineWithPKVsFFI,
        input: StringBatch,
        max_length: i32,
        temperature: f32,
    ) -> FFISlice<'a, UseAsciiStringPattern<'a>> {
        let sampler = RandomSampler::new(temperature);
        s.output_buf = s
            .model
            .generate_batch(input.batch, max_length, &sampler)
            .unwrap_or_default();
        s.output_buf_ffi = s
            .output_buf
            .iter()
            .map(|s| {
                AsciiPointer::from_slice_with_nul(
                    CString::new(s.as_str())
                        .unwrap_or_default()
                        .to_bytes_with_nul(),
                )
                .expect("Failed to convert CString to AsciiPointer")
            })
            .map(|s| UseAsciiStringPattern { ascii_string: s })
            .collect();
        FFISlice::from_slice(s.output_buf_ffi.as_slice())
    }

    #[ffi_service_method(on_panic = "return_default")]
    pub fn generate_argmax_batch(
        s: &'a mut ConditionalGenerationPipelineWithPKVsFFI,
        input: StringBatch,
        max_length: i32,
    ) -> FFISlice<'a, UseAsciiStringPattern<'a>> {
        let sampler = ArgmaxSampler::new();
        s.output_buf = s
            .model
            .generate_batch(input.batch, max_length, &sampler)
            .unwrap();
        s.output_buf_ffi = s
            .output_buf
            .iter()
            .map(|s| {
                AsciiPointer::from_slice_with_nul(
                    CString::new(s.as_str()).unwrap().to_bytes_with_nul(),
                )
                .expect("Failed to convert CString to AsciiPointer")
            })
            .map(|s| UseAsciiStringPattern { ascii_string: s })
            .collect();
        FFISlice::from_slice(s.output_buf_ffi.as_slice())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_generate_topk_sampling() -> Result<()> {
        let e = EnvContainer::new()?;
        let mut pipeline = ConditionalGenerationPipelineWithPKVsFFI::from_pretrained(
            &e,
            AsciiPointer::from_slice_with_nul(CString::new("optimum/gpt2")?.to_bytes_with_nul())?,
            DeviceFFI::CPU,
            GraphOptimizationLevelFFI::Level3,
        )
        .unwrap();

        let output = pipeline.generate_topk_sampling(
            AsciiPointer::from_slice_with_nul(
                CString::new("translate English to French: How old are you?")?.to_bytes_with_nul(),
            )?,
            32,
            5,
            1.0,
        );
        println!(
            "{}",
            output.as_c_str().unwrap().to_string_lossy().to_string()
        );
        Ok(())
    }

    #[test]
    fn test_generate_topk_sampling_batch() -> Result<()> {
        let e = EnvContainer::new()?;
        let mut pipeline = ConditionalGenerationPipelineWithPKVsFFI::from_pretrained(
            &e,
            AsciiPointer::from_slice_with_nul(CString::new("optimum/gpt2")?.to_bytes_with_nul())?,
            DeviceFFI::CPU,
            GraphOptimizationLevelFFI::Level3,
        )
        .unwrap();
        let b = StringBatch {
            batch: vec![
                "translate English to French: How old are you?".to_string(),
                "translate English to French: What is your name?".to_string(),
            ],
        };
        let output = ConditionalGenerationPipelineWithPKVsFFI::generate_topk_sampling_batch(
            &mut pipeline,
            b,
            32,
            5,
            1.0,
        );
        println!("{:?}", output[0].ascii_string.as_c_str().unwrap());
        Ok(())
    }
}
