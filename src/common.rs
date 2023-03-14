use std::collections::HashMap;

use crate::{Error, Result};
use half::{bf16, f16};
use ort::session::{Input, SessionBuilder};
use ort::tensor::{FromArray, InputTensor, TensorElementDataType};
use ort::{ExecutionProvider, GraphOptimizationLevel};

pub enum ORTSession<'a> {
    InMemory(ort::InMemorySession<'a>),
    Owned(ort::Session),
}

pub fn match_to_inputs(
    inputs: &Vec<Input>,
    mut values: HashMap<String, InputTensor>,
) -> Result<Vec<InputTensor>> {
    let mut inputs_array_vector: Vec<InputTensor> = Default::default();
    let input_names = inputs
        .iter()
        .map(|input| input.name.clone())
        .collect::<Vec<String>>();
    // check if inputs contain `.1` and remove it if it won't lead to duplicate inputs
    let input_names = input_names
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
            TensorElementDataType::Float32 => {
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
            TensorElementDataType::Float64 => {
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
            TensorElementDataType::Bool => {
                if let Some(value) = values.remove(input_name) {
                    Ok(value)
                } else {
                    return Err(format!("Missing input: {}", input.name).into());
                }
            }
            TensorElementDataType::Float16 => {
                if let Some(value) = values.remove(input_name) {
                    Ok(cast_input_tensor_f16(value))
                } else {
                    return Err(format!("Missing input: {}", input.name).into());
                }
            }
            TensorElementDataType::Bfloat16 => {
                if let Some(value) = values.remove(input_name) {
                    Ok(cast_input_tensor_bf16(value))
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
            fn [<cast_input_tensor_ $type_>](input: InputTensor) -> InputTensor
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
                    InputTensor::Float16Tensor(array) => { array.mapv(|x| x.to_f32() as $type_) }
                    InputTensor::Bfloat16Tensor(array) => { array.mapv(|x| x.to_f32() as $type_) }
                };
                InputTensor::from_array(array)
            }
        }
    };
}

macro_rules! impl_cast_non_primitive_array {
    ($type_:ty) => {
        ::paste::paste! {
            fn [<cast_input_tensor_ $type_>](input: InputTensor) -> InputTensor
            {
                let array = match input {
                    InputTensor::FloatTensor(array) => { array.mapv(|x| $type_::from_f64(x as f64)) }
                    InputTensor::Uint8Tensor(array) => { array.mapv(|x| $type_::from_f64(x as f64)) }
                    InputTensor::Int8Tensor(array) => { array.mapv(|x| $type_::from_f64(x as f64)) }
                    InputTensor::Uint16Tensor(array) => { array.mapv(|x| $type_::from_f64(x as f64)) }
                    InputTensor::Int16Tensor(array) => { array.mapv(|x| $type_::from_f64(x as f64)) }
                    InputTensor::Int32Tensor(array) => { array.mapv(|x| $type_::from_f64(x as f64)) }
                    InputTensor::Int64Tensor(array) => { array.mapv(|x| $type_::from_f64(x as f64)) }
                    InputTensor::DoubleTensor(array) => { array.mapv(|x| $type_::from_f64(x as f64)) }
                    InputTensor::Uint32Tensor(array) => { array.mapv(|x| $type_::from_f64(x as f64)) }
                    InputTensor::Uint64Tensor(array) => { array.mapv(|x| $type_::from_f64(x as f64)) }
                    InputTensor::StringTensor(array) => { array.mapv(|x| x.parse::<$type_>().unwrap()) }
                    InputTensor::Float16Tensor(array) => { array.mapv(|x| $type_::from_f64(x.to_f64())) }
                    InputTensor::Bfloat16Tensor(array) => { array.mapv(|x| $type_::from_f64(x.to_f64())) }
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
impl_cast_non_primitive_array!(f16);
impl_cast_non_primitive_array!(bf16);

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
        Device::CPU => session_builder
            .with_execution_providers([ExecutionProvider::cpu()])
            .map_err(|e| e.into()),
        #[cfg(feature = "directml")]
        Device::DML => {
            if cfg!(feature = "directml") {
                session_builder
                    .with_execution_providers([ExecutionProvider::directml()])
                    .map_err(|e| e.into())
            } else {
                return Err(Error::GenericError {
                    message: "DML feature is not enabled".to_string(),
                });
            }
        }
        #[cfg(feature = "cuda")]
        Device::CUDA => {
            if cfg!(feature = "cuda") {
                session_builder
                    .with_execution_providers([ExecutionProvider::cuda()])
                    .map_err(|e| e.into())
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
        GraphOptimizationLevel::Disable => GraphOptimizationLevel::Disable,
        GraphOptimizationLevel::Level1 => GraphOptimizationLevel::Level1,
        GraphOptimizationLevel::Level2 => GraphOptimizationLevel::Level2,
        GraphOptimizationLevel::Level3 => GraphOptimizationLevel::Level3,
    }
}
