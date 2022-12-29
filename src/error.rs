use interoptopus::ffi_type;
use onnxruntime::ndarray;
use std::fmt::{Display, Formatter};

pub(crate) type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct Error {
    pub message: String,
}

impl Error {
    pub fn new(message: String) -> Self {
        Self { message }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "TransformersOnnxError: {}", self.message)
    }
}

impl From<onnxruntime::error::OrtError> for Error {
    fn from(err: onnxruntime::error::OrtError) -> Self {
        Self::new(err.to_string())
    }
}

impl From<tokenizers::tokenizer::Error> for Error {
    fn from(err: tokenizers::tokenizer::Error) -> Self {
        Self::new(err.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Self::new(err.to_string())
    }
}

impl From<ndarray::ShapeError> for Error {
    fn from(err: ndarray::ShapeError) -> Self {
        Self::new(err.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Self::new(err.to_string())
    }
}

impl From<interoptopus::Error> for Error {
    fn from(err: interoptopus::Error) -> Self {
        Self::new(err.to_string())
    }
}