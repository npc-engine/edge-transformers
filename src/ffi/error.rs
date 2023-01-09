use interoptopus::ffi_type;

use crate::Error;

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
