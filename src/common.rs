use std::cell::RefMut;
use std::collections::HashMap;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::{env, fs, io};

use onnxruntime::ndarray::{Array, Array2, IxDyn};
use onnxruntime::session::{Input, Output, Session, SessionBuilder};
use onnxruntime::tensor::{FromArray, InputTensor};
use onnxruntime::{GraphOptimizationLevel, TensorElementDataType};
use reqwest::blocking::{get, Client};
use sha2::digest::Update;
use sha2::{Digest, Sha256, Sha512};

use crate::ffi::GraphOptimizationLevelFFI;
use crate::{Error, Result};

pub fn match_to_inputs(
    inputs: &Vec<Input>,
    mut values: HashMap<String, InputTensor<IxDyn>>,
) -> Result<Vec<InputTensor<IxDyn>>> {
    let mut inputs_array_vector: Vec<InputTensor<IxDyn>> = Default::default();
    let input_names = inputs
        .iter()
        .map(|input| input.name.clone())
        .collect::<Vec<String>>();
    // check if inputs contain `.1` and remove it if it won't lead to duplicate inputs
    let mut input_names = input_names
        .iter()
        .map(|input_name| {
            if input_name.ends_with(".1") {
                let input_name = input_name.trim_end_matches(".1");
                if !input_names.contains(&input_name.to_string()) {
                    return input_name.to_string();
                }
            }
            input_name.to_string()
        })
        .collect::<Vec<String>>();
    for (input, input_name) in inputs.iter().zip(input_names.iter()) {
        inputs_array_vector.push(match input.input_type {
            TensorElementDataType::Float => {
                if let Some(value) = values.remove(input_name) {
                    Ok::<_, Error>(cast_input_tensor_f32(value))
                } else {
                    return Err(format!("Missing input: {}", input.name).into());
                }
            }
            TensorElementDataType::Uint8 => {
                if let Some(value) = values.remove(input_name) {
                    Ok(cast_input_tensor_u8(value))
                } else {
                    return Err(format!("Missing input: {}", input.name).into());
                }
            }
            TensorElementDataType::Int8 => {
                if let Some(value) = values.remove(input_name) {
                    Ok(cast_input_tensor_i8(value))
                } else {
                    return Err(format!("Missing input: {}", input.name).into());
                }
            }
            TensorElementDataType::Uint16 => {
                if let Some(value) = values.remove(input_name) {
                    Ok(cast_input_tensor_u16(value))
                } else {
                    return Err(format!("Missing input: {}", input.name).into());
                }
            }
            TensorElementDataType::Int16 => {
                if let Some(value) = values.remove(input_name) {
                    Ok(cast_input_tensor_i16(value))
                } else {
                    return Err(format!("Missing input: {}", input.name).into());
                }
            }
            TensorElementDataType::Int32 => {
                if let Some(value) = values.remove(input_name) {
                    Ok(cast_input_tensor_i32(value))
                } else {
                    return Err(format!("Missing input: {}", input.name).into());
                }
            }
            TensorElementDataType::Int64 => {
                if let Some(value) = values.remove(input_name) {
                    Ok(cast_input_tensor_i64(value))
                } else {
                    return Err(format!("Missing input: {}", input.name).into());
                }
            }
            TensorElementDataType::String => {
                if let Some(value) = values.remove(input_name) {
                    Ok(value)
                } else {
                    return Err(format!("Missing input: {}", input.name).into());
                }
            }
            TensorElementDataType::Double => {
                if let Some(value) = values.remove(input_name) {
                    Ok(cast_input_tensor_f64(value))
                } else {
                    return Err(format!("Missing input: {}", input.name).into());
                }
            }
            TensorElementDataType::Uint32 => {
                if let Some(value) = values.remove(input_name) {
                    Ok(cast_input_tensor_u32(value))
                } else {
                    return Err(format!("Missing input: {}", input.name).into());
                }
            }
            TensorElementDataType::Uint64 => {
                if let Some(value) = values.remove(input_name) {
                    Ok(cast_input_tensor_u64(value))
                } else {
                    return Err(format!("Missing input: {}", input.name).into());
                }
            }
        }?);
    }
    Ok(inputs_array_vector)
}

macro_rules! impl_cast_input_array {
    ($type_:ty) => {
        ::paste::paste! {
            fn [<cast_input_tensor_ $type_>](input: InputTensor<IxDyn>) -> InputTensor<IxDyn>
            {
                let array = match input {
                    InputTensor::FloatTensor(array) => { array.mapv(|x| x as $type_) }
                    InputTensor::Uint8Tensor(array) => { array.mapv(|x| x as $type_) }
                    InputTensor::Int8Tensor(array) => { array.mapv(|x| x as $type_) }
                    InputTensor::Uint16Tensor(array) => { array.mapv(|x| x as $type_) }
                    InputTensor::Int16Tensor(array) => { array.mapv(|x| x as $type_) }
                    InputTensor::Int32Tensor(array) => { array.mapv(|x| x as $type_) }
                    InputTensor::Int64Tensor(array) => { array.mapv(|x| x as $type_) }
                    InputTensor::DoubleTensor(array) => { array.mapv(|x| x as $type_) }
                    InputTensor::Uint32Tensor(array) => { array.mapv(|x| x as $type_) }
                    InputTensor::Uint64Tensor(array) => { array.mapv(|x| x as $type_) }
                    InputTensor::StringTensor(array) => { array.mapv(|x| x.parse::<$type_>().unwrap()) }
                };
                InputTensor::from_array(array)
            }
        }
    };
}

impl_cast_input_array!(f32);
impl_cast_input_array!(u8);
impl_cast_input_array!(i8);
impl_cast_input_array!(u16);
impl_cast_input_array!(i16);
impl_cast_input_array!(i32);
impl_cast_input_array!(i64);
impl_cast_input_array!(f64);
impl_cast_input_array!(u32);
impl_cast_input_array!(u64);

#[derive(Debug, Clone)]
/// Device enum to specify the device to run the model on
pub enum Device {
    CPU,
    #[cfg(feature = "directml")]
    DML,
    #[cfg(feature = "cuda")]
    CUDA,
}

pub fn apply_device(
    session_builder: SessionBuilder,
    device: Device,
) -> std::result::Result<SessionBuilder, Error> {
    match device {
        Device::CPU => session_builder.use_cpu(1).map_err(|e| e.into()),
        #[cfg(feature = "directml")]
        Device::DML => {
            if cfg!(feature = "directml") {
                session_builder.use_dml().map_err(|e| e.into())
            } else {
                return Err(Error::GenericError {
                    message: "DML feature is not enabled".to_string(),
                });
            }
        }
        #[cfg(feature = "cuda")]
        Device::CUDA => {
            if cfg!(feature = "cuda") {
                session_builder.use_cuda(0).map_err(|e| e.into())
            } else {
                return Err(Error::GenericError {
                    message: "CUDA feature is not enabled".to_string(),
                });
            }
        }
    }
}

pub fn clone(opt_level: &GraphOptimizationLevel) -> GraphOptimizationLevel {
    match opt_level {
        GraphOptimizationLevel::DisableAll => GraphOptimizationLevel::DisableAll,
        GraphOptimizationLevel::Basic => GraphOptimizationLevel::Basic,
        GraphOptimizationLevel::Extended => GraphOptimizationLevel::Extended,
        GraphOptimizationLevel::All => GraphOptimizationLevel::All,
    }
}
