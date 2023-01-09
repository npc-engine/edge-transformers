use std::fmt::{Display, Formatter};

use onnxruntime::{ndarray, OrtError};

pub(crate) type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    OnnxIncorrectInputs {
        message: String,
        expected: Vec<String>,
        actual: Vec<String>,
    },
    OnnxIncorrectOutputs {
        message: String,
        expected: Vec<String>,
        actual: Vec<String>,
    },
    OnnxInputOutputMismatch {
        input: Vec<String>,
        output: Vec<String>,
    },
    OrtError {
        error: OrtError,
    },
    TokenizerError {
        error: tokenizers::Error,
    },
    NdarrayError {
        error: ndarray::ShapeError,
    },
    IOError {
        error: std::io::Error,
    },
    SerdeJsonError {
        error: serde_json::Error,
    },
    InteroptopusError {
        error: interoptopus::Error,
    },
    CStringError {
        error: std::ffi::NulError,
    },
    GenericError {
        message: String,
    },
    MissingId2Label,
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::OnnxIncorrectInputs {
                message,
                expected,
                actual,
            } => write!(
                f,
                "Incorrect inputs: {}. Expected: {:?}, Actual: {:?}",
                message, expected, actual
            ),
            Error::OnnxIncorrectOutputs {
                message,
                expected,
                actual,
            } => write!(
                f,
                "Incorrect outputs: {}. Expected: {:?}, Actual: {:?}",
                message, expected, actual
            ),
            Error::OnnxInputOutputMismatch { input, output } => {
                write!(
                    f,
                    "Input and output mismatch. Input: {:?}, output: {:?}",
                    input, output
                )
            }
            Error::OrtError { error } => write!(f, "ONNX Runtime error: {}", error),
            Error::TokenizerError { error } => write!(f, "Tokenizer error: {}", error),
            Error::NdarrayError { error } => write!(f, "Ndarray error: {}", error),
            Error::IOError { error } => write!(f, "IO error: {}", error),
            Error::SerdeJsonError { error } => write!(f, "Serde JSON error: {}", error),
            Error::InteroptopusError { error } => write!(f, "Interoptopus error: {}", error),
            Error::CStringError { error } => write!(f, "CString error: {}", error),
            Error::GenericError { message } => write!(f, "Generic error: {}", message),
            Error::MissingId2Label => write!(f, "Missing id2label in config.json"),
        }
    }
}

impl From<OrtError> for Error {
    fn from(err: OrtError) -> Self {
        Error::OrtError { error: err }
    }
}

impl From<tokenizers::tokenizer::Error> for Error {
    fn from(err: tokenizers::tokenizer::Error) -> Self {
        Error::TokenizerError { error: err }
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::IOError { error: err }
    }
}

impl From<ndarray::ShapeError> for Error {
    fn from(err: ndarray::ShapeError) -> Self {
        Error::NdarrayError { error: err }
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::SerdeJsonError { error: err }
    }
}

impl From<interoptopus::Error> for Error {
    fn from(err: interoptopus::Error) -> Self {
        Error::InteroptopusError { error: err }
    }
}

impl From<std::ffi::NulError> for Error {
    fn from(err: std::ffi::NulError) -> Self {
        Error::CStringError { error: err }
    }
}

impl From<String> for Error {
    fn from(err: String) -> Self {
        Error::GenericError { message: err }
    }
}
