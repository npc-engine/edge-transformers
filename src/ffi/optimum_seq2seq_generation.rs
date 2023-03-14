use std::borrow::Borrow;
use std::ffi::CString;
use std::path::Path;

use interoptopus::patterns::option::FFIOption;
use interoptopus::patterns::slice::FFISlice;
use interoptopus::patterns::string::AsciiPointer;
use interoptopus::{ffi_service, ffi_service_ctor, ffi_service_method, ffi_type};

use crate::error::Result;
use crate::ffi::{
    error::FFIError, DeviceFFI, EnvContainer, GraphOptimizationLevelFFI, StringBatch,
    UseAsciiStringPattern,
};
use crate::sampling::{ArgmaxSampler, RandomSampler, TopKSampler};
use crate::OptimumSeq2SeqPipeline;

#[ffi_type(opaque, name = "OptimumSeq2SeqPipeline")]
pub struct OptimumSeq2SeqPipelineFFI<'a> {
    pub model: OptimumSeq2SeqPipeline<'a>,
    pub output_buf: Vec<String>,
    pub output_buf_ffi: Vec<UseAsciiStringPattern<'a>>,
}

#[ffi_service(error = "FFIError", prefix = "onnx_optimum_seq2seq_")]
impl<'a> OptimumSeq2SeqPipelineFFI<'a> {
    #[ffi_service_ctor]
    pub fn from_pretrained(
        env: &'a EnvContainer,
        model_id: AsciiPointer<'a>,
        device: DeviceFFI,
        optimization: GraphOptimizationLevelFFI,
    ) -> Result<Self> {
        let model = OptimumSeq2SeqPipeline::from_pretrained(
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
    pub fn create_from_memory(
        env: &'a EnvContainer,
        encoder_model: &'a FFISlice<'a, u8>,
        decoder_model: &'a FFISlice<'a, u8>,
        tokenizer_config: AsciiPointer<'a>,
        special_tokens_map: AsciiPointer<'a>,
        device: DeviceFFI,
        optimization: GraphOptimizationLevelFFI,
    ) -> Result<Self> {
        let model = OptimumSeq2SeqPipeline::new_from_memory(
            env.env.clone(),
            encoder_model.as_slice().clone(),
            decoder_model.as_slice().clone(),
            tokenizer_config
                .as_c_str()
                .unwrap()
                .to_string_lossy()
                .to_string(),
            special_tokens_map
                .as_c_str()
                .unwrap()
                .to_string_lossy()
                .to_string(),
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
    pub fn create_from_files(
        env: &'a EnvContainer,
        encoder_model_path: AsciiPointer<'a>,
        decoder_model_path: AsciiPointer<'a>,
        tokenizer_config_path: AsciiPointer<'a>,
        special_tokens_map_path: AsciiPointer<'a>,
        device: DeviceFFI,
        optimization: GraphOptimizationLevelFFI,
    ) -> Result<Self> {
        let model = OptimumSeq2SeqPipeline::new_from_files(
            env.env.clone(),
            Path::new(&*encoder_model_path.as_c_str().unwrap().to_string_lossy()).to_path_buf(),
            Path::new(&*decoder_model_path.as_c_str().unwrap().to_string_lossy()).to_path_buf(),
            Path::new(&*tokenizer_config_path.as_c_str().unwrap().to_string_lossy()).to_path_buf(),
            Path::new(
                &*special_tokens_map_path
                    .as_c_str()
                    .unwrap()
                    .to_string_lossy(),
            )
            .to_path_buf(),
            device.into(),
            optimization.into(),
        )?;
        Ok(Self {
            model,
            output_buf: Vec::new(),
            output_buf_ffi: Vec::new(),
        })
    }

    #[ffi_service_method(on_panic = "return_default")]
    pub fn generate_topk_sampling(
        &mut self,
        input: AsciiPointer,
        decoder_input: FFIOption<AsciiPointer>,
        max_length: i32,
        topk: i32,
        temperature: f32,
    ) -> AsciiPointer<'a> {
        let sampler = TopKSampler::new(topk as usize, temperature);
        let decoder_input = decoder_input
            .into_option()
            .map(|s| s.as_c_str().unwrap().to_string_lossy().to_string());
        let output = self
            .model
            .generate(
                &*input.as_c_str().unwrap().to_string_lossy(),
                decoder_input.as_ref().map(|s| s.borrow()),
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
        decoder_input: FFIOption<AsciiPointer>,
        max_length: i32,
        temperature: f32,
    ) -> AsciiPointer<'a> {
        let sampler = RandomSampler::new(temperature);
        let decoder_input = decoder_input
            .into_option()
            .map(|s| s.as_c_str().unwrap().to_string_lossy().to_string());
        let output = self
            .model
            .generate(
                &*input.as_c_str().unwrap().to_string_lossy(),
                decoder_input.as_ref().map(|s| s.borrow()),
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
        decoder_input: FFIOption<AsciiPointer>,
        max_length: i32,
    ) -> AsciiPointer<'a> {
        let sampler = ArgmaxSampler::new();

        let decoder_input_string = decoder_input
            .into_option()
            .map(|s| s.as_c_str().unwrap().to_string_lossy().to_string());
        let output = self
            .model
            .generate(
                &*input.as_c_str().unwrap().to_string_lossy(),
                decoder_input_string.as_ref().map(|s| &**s),
                max_length,
                &sampler,
            )
            .unwrap();
        AsciiPointer::from_slice_with_nul(CString::new(output).unwrap().to_bytes_with_nul())
            .expect("Failed to convert CString to AsciiPointer")
    }

    #[ffi_service_method(on_panic = "return_default")]
    pub fn generate_topk_sampling_batch(
        s: &'a mut OptimumSeq2SeqPipelineFFI,
        input: StringBatch,
        decoder_input: FFIOption<StringBatch>,
        max_length: i32,
        topk: i32,
        temperature: f32,
    ) -> FFISlice<'a, UseAsciiStringPattern<'a>> {
        let sampler = TopKSampler::new(topk as usize, temperature);
        s.output_buf = s
            .model
            .generate_batch(
                input.batch,
                decoder_input.into_option().map(|s| s.batch),
                max_length,
                &sampler,
            )
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
        s: &'a mut OptimumSeq2SeqPipelineFFI,
        input: StringBatch,
        decoder_input: FFIOption<StringBatch>,
        max_length: i32,
        temperature: f32,
    ) -> FFISlice<'a, UseAsciiStringPattern<'a>> {
        let sampler = RandomSampler::new(temperature);
        s.output_buf = s
            .model
            .generate_batch(
                input.batch,
                decoder_input.into_option().map(|s| s.batch),
                max_length,
                &sampler,
            )
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
    pub fn generate_argmax_batch(
        s: &'a mut OptimumSeq2SeqPipelineFFI,
        input: StringBatch,
        decoder_input: FFIOption<StringBatch>,
        max_length: i32,
    ) -> FFISlice<'a, UseAsciiStringPattern<'a>> {
        let sampler = ArgmaxSampler::new();
        s.output_buf = s
            .model
            .generate_batch(
                input.batch,
                decoder_input.into_option().map(|s| s.batch),
                max_length,
                &sampler,
            )
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
        let mut pipeline = OptimumSeq2SeqPipelineFFI::from_pretrained(
            &e,
            AsciiPointer::from_slice_with_nul(
                CString::new("optimum/t5-small")?.to_bytes_with_nul(),
            )?,
            DeviceFFI::CPU,
            GraphOptimizationLevelFFI::Level3,
        )
        .unwrap();

        let output = pipeline.generate_topk_sampling(
            AsciiPointer::from_slice_with_nul(
                CString::new("translate English to French: How old are you?")?.to_bytes_with_nul(),
            )?,
            FFIOption::some(AsciiPointer::from_slice_with_nul(b"<pad>\0").unwrap()),
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
        let mut pipeline = OptimumSeq2SeqPipelineFFI::from_pretrained(
            &e,
            AsciiPointer::from_slice_with_nul(
                CString::new("optimum/t5-small")?.to_bytes_with_nul(),
            )?,
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
        let b_dec = StringBatch {
            batch: vec!["<pad>".to_string(), "<pad>".to_string()],
        };
        let output = OptimumSeq2SeqPipelineFFI::generate_topk_sampling_batch(
            &mut pipeline,
            b,
            FFIOption::some(b_dec),
            32,
            5,
            1.0,
        );
        println!(
            "{:?}",
            output.as_slice()[0]
                .ascii_string
                .as_c_str()
                .unwrap()
                .to_string_lossy()
                .to_string()
        );
        println!(
            "{:?}",
            output.as_slice()[1]
                .ascii_string
                .as_c_str()
                .unwrap()
                .to_string_lossy()
                .to_string()
        );
        Ok(())
    }
}
