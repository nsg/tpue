mod raw;

use std::ffi::{CStr, CString, c_void};
use std::fmt;
use std::marker::PhantomData;
use std::path::Path;
use std::ptr::{self, NonNull};
use std::str::FromStr;
use std::sync::Arc;

use raw::{Api, EdgeTpuDeviceType};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("could not load shared library {path}: {source}")]
    LoadLibrary {
        path: String,
        #[source]
        source: libloading::Error,
    },
    #[error("could not load required symbol {name}: {source}")]
    LoadSymbol {
        name: &'static str,
        #[source]
        source: libloading::Error,
    },
    #[error("path contains a NUL byte: {0}")]
    PathContainsNul(String),
    #[error("TensorFlow Lite could not create model from {0}")]
    ModelCreate(String),
    #[error("TensorFlow Lite could not create interpreter options")]
    OptionsCreate,
    #[error("Edge TPU device list was null despite containing {0} entries")]
    NullDeviceList(usize),
    #[error("Edge TPU device {0} has a null path")]
    NullDevicePath(usize),
    #[error("no Edge TPU device matches {0}")]
    DeviceNotFound(Device),
    #[error("Edge TPU could not create a delegate for {0}")]
    DelegateCreate(Device),
    #[error("TensorFlow Lite could not create an interpreter")]
    InterpreterCreate,
    #[error("TensorFlow Lite {operation} failed with status {status}")]
    Status {
        operation: &'static str,
        status: i32,
    },
    #[error("TensorFlow Lite returned a negative {kind} tensor count: {count}")]
    NegativeTensorCount { kind: &'static str, count: i32 },
    #[error("{kind} tensor index {index} is out of range for {count} tensors")]
    TensorIndex {
        kind: &'static str,
        index: usize,
        count: usize,
    },
    #[error("TensorFlow Lite returned a null {kind} tensor at index {index}")]
    NullTensor { kind: &'static str, index: usize },
    #[error("TensorFlow Lite returned a negative tensor dimension count: {0}")]
    NegativeDimensionCount(i32),
    #[error("TensorFlow Lite returned a negative tensor dimension at index {index}: {value}")]
    NegativeDimension { index: usize, value: i32 },
    #[error("tensor data is unavailable for a {0}-byte tensor")]
    NullTensorData(usize),
    #[error("tensor buffer has {actual} bytes, expected {expected}")]
    BufferSize { expected: usize, actual: usize },
    #[error("invalid Edge TPU device {0:?}; expected usb, usb:0, pci, or pci:0")]
    InvalidDevice(String),
    #[error("Edge TPU runtime returned a null version string")]
    NullVersion,
}

#[derive(Clone)]
pub struct Runtime {
    api: Arc<Api>,
}

impl Runtime {
    pub fn load(tflite_library: &str, edgetpu_library: &str) -> Result<Self> {
        Ok(Self {
            api: Arc::new(Api::load(tflite_library, edgetpu_library)?),
        })
    }

    pub fn version(&self) -> Result<String> {
        // SAFETY: The function pointer is loaded from libedgetpu and its returned pointer is
        // documented as a borrowed, null-terminated runtime version string.
        let version = unsafe { (self.api.edgetpu_version)() };
        if version.is_null() {
            return Err(Error::NullVersion);
        }
        // SAFETY: A non-null edgetpu_version result points to a null-terminated string owned by
        // the library, which remains loaded through self.api.
        Ok(unsafe { CStr::from_ptr(version) }
            .to_string_lossy()
            .into_owned())
    }

    pub fn devices(&self) -> Result<Vec<DeviceInfo>> {
        let devices = DeviceList::new(&self.api)?;
        devices.copy()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceKind {
    Usb,
    Pci,
}

impl fmt::Display for DeviceKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Usb => "usb",
            Self::Pci => "pci",
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceInfo {
    pub kind: DeviceKind,
    pub path: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Device {
    Usb(Option<usize>),
    Pci(Option<usize>),
}

impl Device {
    fn kind(self) -> DeviceKind {
        match self {
            Self::Usb(_) => DeviceKind::Usb,
            Self::Pci(_) => DeviceKind::Pci,
        }
    }

    fn index(self) -> Option<usize> {
        match self {
            Self::Usb(index) | Self::Pci(index) => index,
        }
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::Usb(None)
    }
}

impl fmt::Display for Device {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.kind())?;
        if let Some(index) = self.index() {
            write!(formatter, ":{index}")?;
        }
        Ok(())
    }
}

impl FromStr for Device {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "usb" => Ok(Self::Usb(None)),
            "usb:0" => Ok(Self::Usb(Some(0))),
            "pci" => Ok(Self::Pci(None)),
            "pci:0" => Ok(Self::Pci(Some(0))),
            other => Err(Error::InvalidDevice(other.to_owned())),
        }
    }
}

pub struct Model {
    raw: NonNull<raw::TfLiteModel>,
    api: Arc<Api>,
}

impl Model {
    pub fn load(runtime: &Runtime, path: impl AsRef<Path>) -> Result<Self> {
        let display_path = path.as_ref().display().to_string();
        let c_path = CString::new(display_path.as_bytes())
            .map_err(|_| Error::PathContainsNul(display_path.clone()))?;
        // SAFETY: c_path is null-terminated and valid for the call. The returned model is owned
        // by the caller and paired with TfLiteModelDelete in Drop.
        let raw = unsafe { (runtime.api.model_create_from_file)(c_path.as_ptr()) };
        let raw = NonNull::new(raw).ok_or(Error::ModelCreate(display_path))?;
        Ok(Self {
            raw,
            api: Arc::clone(&runtime.api),
        })
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // SAFETY: raw was returned by TfLiteModelCreateFromFile, is uniquely owned here, and is
        // deleted exactly once while the library remains loaded through self.api.
        unsafe { (self.api.model_delete)(self.raw.as_ptr()) };
    }
}

struct Options {
    raw: NonNull<raw::TfLiteInterpreterOptions>,
    api: Arc<Api>,
}

impl Options {
    fn new(api: &Arc<Api>) -> Result<Self> {
        // SAFETY: The loaded function takes no arguments and returns an owned options pointer.
        let raw = unsafe { (api.options_create)() };
        Ok(Self {
            raw: NonNull::new(raw).ok_or(Error::OptionsCreate)?,
            api: Arc::clone(api),
        })
    }
}

impl Drop for Options {
    fn drop(&mut self) {
        // SAFETY: raw is uniquely owned and was created by TfLiteInterpreterOptionsCreate.
        unsafe { (self.api.options_delete)(self.raw.as_ptr()) };
    }
}

struct EdgeDelegate {
    raw: NonNull<raw::TfLiteDelegate>,
    api: Arc<Api>,
}

impl EdgeDelegate {
    fn new(api: &Arc<Api>, device: Device) -> Result<Self> {
        let path = match device.index() {
            Some(index) => Some(DeviceList::new(api)?.path_for(device.kind(), index)?),
            None => None,
        };
        let path_pointer = path.as_ref().map_or(ptr::null(), |path| path.as_ptr());
        let device_type = match device.kind() {
            DeviceKind::Usb => EdgeTpuDeviceType::Usb,
            DeviceKind::Pci => EdgeTpuDeviceType::Pci,
        };
        // SAFETY: device_type is a valid libedgetpu enum and path_pointer is either null or a
        // live null-terminated device path. No delegate options are supplied.
        let raw =
            unsafe { (api.edgetpu_create_delegate)(device_type, path_pointer, ptr::null(), 0) };
        Ok(Self {
            raw: NonNull::new(raw).ok_or(Error::DelegateCreate(device))?,
            api: Arc::clone(api),
        })
    }
}

impl Drop for EdgeDelegate {
    fn drop(&mut self) {
        // SAFETY: raw is uniquely owned and was returned by edgetpu_create_delegate.
        unsafe { (self.api.edgetpu_free_delegate)(self.raw.as_ptr()) };
    }
}

struct DeviceList<'a> {
    raw: *mut raw::EdgeTpuDevice,
    len: usize,
    api: &'a Api,
}

impl<'a> DeviceList<'a> {
    fn new(api: &'a Api) -> Result<Self> {
        let mut len = 0;
        // SAFETY: &mut len is valid for the out parameter and any returned allocation is paired
        // with edgetpu_free_devices in Drop.
        let raw = unsafe { (api.edgetpu_list_devices)(&mut len) };
        if len > 0 && raw.is_null() {
            return Err(Error::NullDeviceList(len));
        }
        Ok(Self { raw, len, api })
    }

    fn copy(&self) -> Result<Vec<DeviceInfo>> {
        let mut result = Vec::with_capacity(self.len);
        for index in 0..self.len {
            let device = self.get(index);
            if device.path.is_null() {
                return Err(Error::NullDevicePath(index));
            }
            // SAFETY: edgetpu_list_devices returned an array of len records whose path fields are
            // null-terminated strings valid until edgetpu_free_devices.
            let path = unsafe { CStr::from_ptr(device.path) }
                .to_string_lossy()
                .into_owned();
            result.push(DeviceInfo {
                kind: kind_from_raw(device.device_type),
                path,
            });
        }
        Ok(result)
    }

    fn path_for(&self, kind: DeviceKind, wanted_index: usize) -> Result<CString> {
        let mut matching_index = 0;
        for index in 0..self.len {
            let device = self.get(index);
            if kind_from_raw(device.device_type) != kind {
                continue;
            }
            if matching_index == wanted_index {
                if device.path.is_null() {
                    return Err(Error::NullDevicePath(index));
                }
                // SAFETY: The device record comes from the live list and its non-null path is a
                // valid null-terminated string until this list is freed.
                return Ok(unsafe { CStr::from_ptr(device.path) }.to_owned());
            }
            matching_index += 1;
        }
        let requested = match kind {
            DeviceKind::Usb => Device::Usb(Some(wanted_index)),
            DeviceKind::Pci => Device::Pci(Some(wanted_index)),
        };
        Err(Error::DeviceNotFound(requested))
    }

    fn get(&self, index: usize) -> &raw::EdgeTpuDevice {
        // SAFETY: Every caller supplies an index in 0..self.len, and a positive-length list was
        // checked non-null at construction.
        unsafe { &*self.raw.add(index) }
    }
}

impl Drop for DeviceList<'_> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: raw is the allocation returned by edgetpu_list_devices and is freed once.
            unsafe { (self.api.edgetpu_free_devices)(self.raw) };
        }
    }
}

fn kind_from_raw(kind: EdgeTpuDeviceType) -> DeviceKind {
    match kind {
        EdgeTpuDeviceType::Usb => DeviceKind::Usb,
        EdgeTpuDeviceType::Pci => DeviceKind::Pci,
    }
}

pub struct Interpreter {
    raw: NonNull<raw::TfLiteInterpreter>,
    delegate: EdgeDelegate,
    _model: Model,
}

impl Interpreter {
    pub fn new(model: Model, device: Device, num_threads: i32) -> Result<Self> {
        let options = Options::new(&model.api)?;
        let delegate = EdgeDelegate::new(&model.api, device)?;
        // SAFETY: options and delegate are valid, and the delegate remains alive in the returned
        // Interpreter for at least as long as the native interpreter.
        unsafe {
            (model.api.options_set_num_threads)(options.raw.as_ptr(), num_threads);
            (model.api.options_add_delegate)(options.raw.as_ptr(), delegate.raw.as_ptr());
        }
        // SAFETY: model and options are live valid objects. The model is retained by the wrapper,
        // while options may be dropped immediately after interpreter creation per the C API.
        let raw =
            unsafe { (model.api.interpreter_create)(model.raw.as_ptr(), options.raw.as_ptr()) };
        let raw = NonNull::new(raw).ok_or(Error::InterpreterCreate)?;
        // SAFETY: raw is a newly created interpreter that has not yet exposed tensor pointers.
        let status = unsafe { (model.api.interpreter_allocate_tensors)(raw.as_ptr()) };
        if status != 0 {
            // SAFETY: raw is owned locally and must be deleted on allocation failure.
            unsafe { (model.api.interpreter_delete)(raw.as_ptr()) };
            return Err(Error::Status {
                operation: "tensor allocation",
                status,
            });
        }
        Ok(Self {
            raw,
            delegate,
            _model: model,
        })
    }

    pub fn input_count(&self) -> Result<usize> {
        // SAFETY: self.raw is a live interpreter pointer.
        let count =
            unsafe { (self.delegate.api.interpreter_get_input_tensor_count)(self.raw.as_ptr()) };
        tensor_count("input", count)
    }

    pub fn output_count(&self) -> Result<usize> {
        // SAFETY: self.raw is a live interpreter pointer.
        let count =
            unsafe { (self.delegate.api.interpreter_get_output_tensor_count)(self.raw.as_ptr()) };
        tensor_count("output", count)
    }

    pub fn input(&self, index: usize) -> Result<Tensor<'_>> {
        let count = self.input_count()?;
        check_index("input", index, count)?;
        // SAFETY: index was checked against the native input tensor count and self.raw is live.
        let raw = unsafe {
            (self.delegate.api.interpreter_get_input_tensor)(self.raw.as_ptr(), index as i32)
        };
        Ok(Tensor::new(
            NonNull::new(raw).ok_or(Error::NullTensor {
                kind: "input",
                index,
            })?,
            &self.delegate.api,
        ))
    }

    pub fn input_mut(&mut self, index: usize) -> Result<TensorMut<'_>> {
        let count = self.input_count()?;
        check_index("input", index, count)?;
        // SAFETY: index was checked against the native input tensor count, self.raw is live, and
        // the exclusive interpreter borrow prevents aliasing another mutable tensor view.
        let raw = unsafe {
            (self.delegate.api.interpreter_get_input_tensor)(self.raw.as_ptr(), index as i32)
        };
        Ok(TensorMut::new(
            NonNull::new(raw).ok_or(Error::NullTensor {
                kind: "input",
                index,
            })?,
            &self.delegate.api,
        ))
    }

    pub fn output(&self, index: usize) -> Result<Tensor<'_>> {
        let count = self.output_count()?;
        check_index("output", index, count)?;
        // SAFETY: index was checked against the native output tensor count and self.raw is live.
        let raw = unsafe {
            (self.delegate.api.interpreter_get_output_tensor)(self.raw.as_ptr(), index as i32)
        };
        Ok(Tensor::new(
            NonNull::new(raw.cast_mut()).ok_or(Error::NullTensor {
                kind: "output",
                index,
            })?,
            &self.delegate.api,
        ))
    }

    pub fn write_input(&mut self, index: usize, data: &[u8]) -> Result<()> {
        self.input_mut(index)?.copy_from(data)
    }

    pub fn read_output(&self, index: usize) -> Result<Vec<u8>> {
        self.output(index)?.to_vec()
    }

    pub fn invoke(&mut self) -> Result<()> {
        // SAFETY: The interpreter is fully allocated and the exclusive borrow prevents tensor
        // access or another invocation while native inference runs.
        let status = unsafe { (self.delegate.api.interpreter_invoke)(self.raw.as_ptr()) };
        status_result("inference", status)
    }
}

impl Drop for Interpreter {
    fn drop(&mut self) {
        // SAFETY: raw is uniquely owned and deleted before the delegate and loaded libraries are
        // dropped from this wrapper.
        unsafe { (self.delegate.api.interpreter_delete)(self.raw.as_ptr()) };
    }
}

fn tensor_count(kind: &'static str, count: i32) -> Result<usize> {
    usize::try_from(count).map_err(|_| Error::NegativeTensorCount { kind, count })
}

fn check_index(kind: &'static str, index: usize, count: usize) -> Result<()> {
    if index >= count {
        return Err(Error::TensorIndex { kind, index, count });
    }
    Ok(())
}

fn status_result(operation: &'static str, status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::Status { operation, status })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ElementType {
    None,
    Float32,
    Int32,
    UInt8,
    Int64,
    String,
    Bool,
    Int16,
    Complex64,
    Int8,
    Float16,
    Float64,
    Complex128,
    UInt64,
    Resource,
    Variant,
    UInt32,
    UInt16,
    Int4,
    BFloat16,
    Unknown(i32),
}

impl From<i32> for ElementType {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::None,
            1 => Self::Float32,
            2 => Self::Int32,
            3 => Self::UInt8,
            4 => Self::Int64,
            5 => Self::String,
            6 => Self::Bool,
            7 => Self::Int16,
            8 => Self::Complex64,
            9 => Self::Int8,
            10 => Self::Float16,
            11 => Self::Float64,
            12 => Self::Complex128,
            13 => Self::UInt64,
            14 => Self::Resource,
            15 => Self::Variant,
            16 => Self::UInt32,
            17 => Self::UInt16,
            18 => Self::Int4,
            19 => Self::BFloat16,
            other => Self::Unknown(other),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quantization {
    pub scale: f32,
    pub zero_point: i32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    pub name: Option<String>,
    pub element_type: ElementType,
    pub shape: Vec<usize>,
    pub byte_size: usize,
    pub quantization: Quantization,
}

pub struct Tensor<'a> {
    raw: NonNull<raw::TfLiteTensor>,
    api: &'a Api,
    _interpreter: PhantomData<&'a Interpreter>,
}

impl<'a> Tensor<'a> {
    fn new(raw: NonNull<raw::TfLiteTensor>, api: &'a Api) -> Self {
        Self {
            raw,
            api,
            _interpreter: PhantomData,
        }
    }

    pub fn element_type(&self) -> ElementType {
        // SAFETY: self.raw is a live tensor borrowed from its interpreter.
        unsafe { (self.api.tensor_type)(self.raw.as_ptr()) }.into()
    }

    pub fn shape(&self) -> Result<Vec<usize>> {
        // SAFETY: self.raw is a live tensor borrowed from its interpreter.
        let count = unsafe { (self.api.tensor_num_dims)(self.raw.as_ptr()) };
        let count = usize::try_from(count).map_err(|_| Error::NegativeDimensionCount(count))?;
        let mut shape = Vec::with_capacity(count);
        for index in 0..count {
            // SAFETY: index is within the dimension count returned by the same live tensor.
            let value = unsafe { (self.api.tensor_dim)(self.raw.as_ptr(), index as i32) };
            shape.push(
                usize::try_from(value).map_err(|_| Error::NegativeDimension { index, value })?,
            );
        }
        Ok(shape)
    }

    pub fn byte_size(&self) -> usize {
        // SAFETY: self.raw is a live tensor borrowed from its interpreter.
        unsafe { (self.api.tensor_byte_size)(self.raw.as_ptr()) }
    }

    pub fn name(&self) -> Option<String> {
        // SAFETY: self.raw is a live tensor borrowed from its interpreter.
        let name = unsafe { (self.api.tensor_name)(self.raw.as_ptr()) };
        if name.is_null() {
            return None;
        }
        // SAFETY: A non-null TfLiteTensorName result is a tensor-owned null-terminated string.
        Some(
            unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned(),
        )
    }

    pub fn quantization(&self) -> Quantization {
        // SAFETY: self.raw is a live tensor borrowed from its interpreter and the returned struct
        // has the exact repr(C) layout from tflite_types.h.
        let params = unsafe { (self.api.tensor_quantization_params)(self.raw.as_ptr()) };
        Quantization {
            scale: params.scale,
            zero_point: params.zero_point,
        }
    }

    pub fn info(&self) -> Result<TensorInfo> {
        Ok(TensorInfo {
            name: self.name(),
            element_type: self.element_type(),
            shape: self.shape()?,
            byte_size: self.byte_size(),
            quantization: self.quantization(),
        })
    }

    pub fn as_bytes(&self) -> Result<&[u8]> {
        let len = self.byte_size();
        // SAFETY: self.raw is a live allocated tensor; the returned buffer remains borrowed for
        // the tensor view lifetime and cannot overlap a mutable interpreter operation.
        let data = unsafe { (self.api.tensor_data)(self.raw.as_ptr()) }.cast::<u8>();
        if data.is_null() && len != 0 {
            return Err(Error::NullTensorData(len));
        }
        if len == 0 {
            return Ok(&[]);
        }
        // SAFETY: TensorFlow Lite reports that data points to byte_size initialized bytes, and
        // the interpreter borrow prevents mutation for the returned slice lifetime.
        Ok(unsafe { std::slice::from_raw_parts(data, len) })
    }

    pub fn copy_to(&self, output: &mut [u8]) -> Result<()> {
        let expected = self.byte_size();
        if output.len() != expected {
            return Err(Error::BufferSize {
                expected,
                actual: output.len(),
            });
        }
        // SAFETY: output is writable for exactly expected bytes, which equals the tensor byte
        // size required by TfLiteTensorCopyToBuffer.
        let status = unsafe {
            (self.api.tensor_copy_to_buffer)(
                self.raw.as_ptr(),
                output.as_mut_ptr().cast::<c_void>(),
                output.len(),
            )
        };
        status_result("tensor copy to buffer", status)
    }

    pub fn to_vec(&self) -> Result<Vec<u8>> {
        let mut data = vec![0; self.byte_size()];
        self.copy_to(&mut data)?;
        Ok(data)
    }

    pub fn to_i8_vec(&self) -> Result<Vec<i8>> {
        let mut data = vec![0; self.byte_size()];
        // SAFETY: i8 has the same one-byte size and alignment as u8. The output is writable for
        // exactly the tensor byte size required by TfLiteTensorCopyToBuffer.
        let status = unsafe {
            (self.api.tensor_copy_to_buffer)(
                self.raw.as_ptr(),
                data.as_mut_ptr().cast::<c_void>(),
                data.len(),
            )
        };
        status_result("tensor copy to buffer", status)?;
        Ok(data)
    }
}

pub struct TensorMut<'a> {
    tensor: Tensor<'a>,
    _exclusive: PhantomData<&'a mut Interpreter>,
}

impl<'a> TensorMut<'a> {
    fn new(raw: NonNull<raw::TfLiteTensor>, api: &'a Api) -> Self {
        Self {
            tensor: Tensor::new(raw, api),
            _exclusive: PhantomData,
        }
    }

    pub fn info(&self) -> Result<TensorInfo> {
        self.tensor.info()
    }

    pub fn as_bytes(&self) -> Result<&[u8]> {
        self.tensor.as_bytes()
    }

    pub fn as_bytes_mut(&mut self) -> Result<&mut [u8]> {
        let len = self.tensor.byte_size();
        // SAFETY: The exclusive interpreter borrow guarantees unique access to this tensor data,
        // which TensorFlow Lite reports as a byte_size-byte allocated buffer.
        let data = unsafe { (self.tensor.api.tensor_data)(self.tensor.raw.as_ptr()) }.cast::<u8>();
        if data.is_null() && len != 0 {
            return Err(Error::NullTensorData(len));
        }
        if len == 0 {
            return Ok(&mut []);
        }
        // SAFETY: data points to len writable tensor bytes and TensorMut's exclusive lifetime
        // prevents any competing access through the interpreter.
        Ok(unsafe { std::slice::from_raw_parts_mut(data, len) })
    }

    pub fn copy_from(&mut self, input: &[u8]) -> Result<()> {
        let expected = self.tensor.byte_size();
        if input.len() != expected {
            return Err(Error::BufferSize {
                expected,
                actual: input.len(),
            });
        }
        // SAFETY: input is readable for exactly expected bytes, which equals the tensor byte size
        // required by TfLiteTensorCopyFromBuffer; TensorMut provides exclusive tensor access.
        let status = unsafe {
            (self.tensor.api.tensor_copy_from_buffer)(
                self.tensor.raw.as_ptr(),
                input.as_ptr().cast::<c_void>(),
                input.len(),
            )
        };
        status_result("tensor copy from buffer", status)
    }
}
