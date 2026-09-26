use std::ffi::{c_char, c_int, c_void};

use libloading::Library;

use super::{Error, Result};

#[repr(C)]
pub(super) struct TfLiteModel {
    _private: [u8; 0],
}

#[repr(C)]
pub(super) struct TfLiteInterpreterOptions {
    _private: [u8; 0],
}

#[repr(C)]
pub(super) struct TfLiteInterpreter {
    _private: [u8; 0],
}

#[repr(C)]
pub(super) struct TfLiteTensor {
    _private: [u8; 0],
}

#[repr(C)]
pub(super) struct TfLiteDelegate {
    _private: [u8; 0],
}

#[derive(Clone, Copy)]
#[repr(C)]
pub(super) struct TfLiteQuantizationParams {
    pub(super) scale: f32,
    pub(super) zero_point: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub(super) enum EdgeTpuDeviceType {
    Pci = 0,
    Usb = 1,
}

#[repr(C)]
pub(super) struct EdgeTpuDevice {
    pub(super) device_type: EdgeTpuDeviceType,
    pub(super) path: *const c_char,
}

pub(super) type ModelCreateFromFile = unsafe extern "C" fn(*const c_char) -> *mut TfLiteModel;
pub(super) type ModelDelete = unsafe extern "C" fn(*mut TfLiteModel);
pub(super) type OptionsCreate = unsafe extern "C" fn() -> *mut TfLiteInterpreterOptions;
pub(super) type OptionsDelete = unsafe extern "C" fn(*mut TfLiteInterpreterOptions);
pub(super) type OptionsAddDelegate =
    unsafe extern "C" fn(*mut TfLiteInterpreterOptions, *mut TfLiteDelegate);
pub(super) type OptionsSetNumThreads = unsafe extern "C" fn(*mut TfLiteInterpreterOptions, i32);
pub(super) type InterpreterCreate = unsafe extern "C" fn(
    *const TfLiteModel,
    *const TfLiteInterpreterOptions,
) -> *mut TfLiteInterpreter;
pub(super) type InterpreterDelete = unsafe extern "C" fn(*mut TfLiteInterpreter);
pub(super) type InterpreterAllocateTensors = unsafe extern "C" fn(*mut TfLiteInterpreter) -> c_int;
pub(super) type InterpreterInvoke = unsafe extern "C" fn(*mut TfLiteInterpreter) -> c_int;
pub(super) type InterpreterGetTensorCount = unsafe extern "C" fn(*const TfLiteInterpreter) -> i32;
pub(super) type InterpreterGetInputTensor =
    unsafe extern "C" fn(*const TfLiteInterpreter, i32) -> *mut TfLiteTensor;
pub(super) type InterpreterGetOutputTensor =
    unsafe extern "C" fn(*const TfLiteInterpreter, i32) -> *const TfLiteTensor;
pub(super) type TensorType = unsafe extern "C" fn(*const TfLiteTensor) -> c_int;
pub(super) type TensorNumDims = unsafe extern "C" fn(*const TfLiteTensor) -> i32;
pub(super) type TensorDim = unsafe extern "C" fn(*const TfLiteTensor, i32) -> i32;
pub(super) type TensorByteSize = unsafe extern "C" fn(*const TfLiteTensor) -> usize;
pub(super) type TensorData = unsafe extern "C" fn(*const TfLiteTensor) -> *mut c_void;
pub(super) type TensorName = unsafe extern "C" fn(*const TfLiteTensor) -> *const c_char;
pub(super) type TensorQuantizationParams =
    unsafe extern "C" fn(*const TfLiteTensor) -> TfLiteQuantizationParams;
pub(super) type TensorCopyFromBuffer =
    unsafe extern "C" fn(*mut TfLiteTensor, *const c_void, usize) -> c_int;
pub(super) type TensorCopyToBuffer =
    unsafe extern "C" fn(*const TfLiteTensor, *mut c_void, usize) -> c_int;

pub(super) type EdgeTpuListDevices = unsafe extern "C" fn(*mut usize) -> *mut EdgeTpuDevice;
pub(super) type EdgeTpuFreeDevices = unsafe extern "C" fn(*mut EdgeTpuDevice);
pub(super) type EdgeTpuCreateDelegate = unsafe extern "C" fn(
    EdgeTpuDeviceType,
    *const c_char,
    *const c_void,
    usize,
) -> *mut TfLiteDelegate;
pub(super) type EdgeTpuFreeDelegate = unsafe extern "C" fn(*mut TfLiteDelegate);
pub(super) type EdgeTpuVersion = unsafe extern "C" fn() -> *const c_char;

pub(super) struct Api {
    _tflite_library: Library,
    _edgetpu_library: Library,
    pub(super) model_create_from_file: ModelCreateFromFile,
    pub(super) model_delete: ModelDelete,
    pub(super) options_create: OptionsCreate,
    pub(super) options_delete: OptionsDelete,
    pub(super) options_add_delegate: OptionsAddDelegate,
    pub(super) options_set_num_threads: OptionsSetNumThreads,
    pub(super) interpreter_create: InterpreterCreate,
    pub(super) interpreter_delete: InterpreterDelete,
    pub(super) interpreter_allocate_tensors: InterpreterAllocateTensors,
    pub(super) interpreter_invoke: InterpreterInvoke,
    pub(super) interpreter_get_input_tensor_count: InterpreterGetTensorCount,
    pub(super) interpreter_get_input_tensor: InterpreterGetInputTensor,
    pub(super) interpreter_get_output_tensor_count: InterpreterGetTensorCount,
    pub(super) interpreter_get_output_tensor: InterpreterGetOutputTensor,
    pub(super) tensor_type: TensorType,
    pub(super) tensor_num_dims: TensorNumDims,
    pub(super) tensor_dim: TensorDim,
    pub(super) tensor_byte_size: TensorByteSize,
    pub(super) tensor_data: TensorData,
    pub(super) tensor_name: TensorName,
    pub(super) tensor_quantization_params: TensorQuantizationParams,
    pub(super) tensor_copy_from_buffer: TensorCopyFromBuffer,
    pub(super) tensor_copy_to_buffer: TensorCopyToBuffer,
    pub(super) edgetpu_list_devices: EdgeTpuListDevices,
    pub(super) edgetpu_free_devices: EdgeTpuFreeDevices,
    pub(super) edgetpu_create_delegate: EdgeTpuCreateDelegate,
    pub(super) edgetpu_free_delegate: EdgeTpuFreeDelegate,
    pub(super) edgetpu_version: EdgeTpuVersion,
}

impl Api {
    pub(super) fn load(tflite_path: &str, edgetpu_path: &str) -> Result<Self> {
        // SAFETY: Loading a caller-selected shared library is the purpose of this module; all
        // resolved symbols are retained only while the owning Library values remain alive.
        let tflite_library =
            unsafe { Library::new(tflite_path) }.map_err(|source| Error::LoadLibrary {
                path: tflite_path.to_owned(),
                source,
            })?;
        // SAFETY: The same lifetime invariant as above applies to the Edge TPU library.
        let edgetpu_library =
            unsafe { Library::new(edgetpu_path) }.map_err(|source| Error::LoadLibrary {
                path: edgetpu_path.to_owned(),
                source,
            })?;

        macro_rules! symbol {
            ($library:expr, $name:literal, $ty:ty) => {{
                // SAFETY: The symbol names and function pointer signatures are copied from the
                // TensorFlow Lite 2.19.1 and libedgetpu C headers. The Library is stored in Api.
                let loaded = unsafe { $library.get::<$ty>(concat!($name, "\0").as_bytes()) }
                    .map_err(|source| Error::LoadSymbol {
                        name: $name,
                        source,
                    })?;
                *loaded
            }};
        }

        let api = Self {
            model_create_from_file: symbol!(
                tflite_library,
                "TfLiteModelCreateFromFile",
                ModelCreateFromFile
            ),
            model_delete: symbol!(tflite_library, "TfLiteModelDelete", ModelDelete),
            options_create: symbol!(
                tflite_library,
                "TfLiteInterpreterOptionsCreate",
                OptionsCreate
            ),
            options_delete: symbol!(
                tflite_library,
                "TfLiteInterpreterOptionsDelete",
                OptionsDelete
            ),
            options_add_delegate: symbol!(
                tflite_library,
                "TfLiteInterpreterOptionsAddDelegate",
                OptionsAddDelegate
            ),
            options_set_num_threads: symbol!(
                tflite_library,
                "TfLiteInterpreterOptionsSetNumThreads",
                OptionsSetNumThreads
            ),
            interpreter_create: symbol!(
                tflite_library,
                "TfLiteInterpreterCreate",
                InterpreterCreate
            ),
            interpreter_delete: symbol!(
                tflite_library,
                "TfLiteInterpreterDelete",
                InterpreterDelete
            ),
            interpreter_allocate_tensors: symbol!(
                tflite_library,
                "TfLiteInterpreterAllocateTensors",
                InterpreterAllocateTensors
            ),
            interpreter_invoke: symbol!(
                tflite_library,
                "TfLiteInterpreterInvoke",
                InterpreterInvoke
            ),
            interpreter_get_input_tensor_count: symbol!(
                tflite_library,
                "TfLiteInterpreterGetInputTensorCount",
                InterpreterGetTensorCount
            ),
            interpreter_get_input_tensor: symbol!(
                tflite_library,
                "TfLiteInterpreterGetInputTensor",
                InterpreterGetInputTensor
            ),
            interpreter_get_output_tensor_count: symbol!(
                tflite_library,
                "TfLiteInterpreterGetOutputTensorCount",
                InterpreterGetTensorCount
            ),
            interpreter_get_output_tensor: symbol!(
                tflite_library,
                "TfLiteInterpreterGetOutputTensor",
                InterpreterGetOutputTensor
            ),
            tensor_type: symbol!(tflite_library, "TfLiteTensorType", TensorType),
            tensor_num_dims: symbol!(tflite_library, "TfLiteTensorNumDims", TensorNumDims),
            tensor_dim: symbol!(tflite_library, "TfLiteTensorDim", TensorDim),
            tensor_byte_size: symbol!(tflite_library, "TfLiteTensorByteSize", TensorByteSize),
            tensor_data: symbol!(tflite_library, "TfLiteTensorData", TensorData),
            tensor_name: symbol!(tflite_library, "TfLiteTensorName", TensorName),
            tensor_quantization_params: symbol!(
                tflite_library,
                "TfLiteTensorQuantizationParams",
                TensorQuantizationParams
            ),
            tensor_copy_from_buffer: symbol!(
                tflite_library,
                "TfLiteTensorCopyFromBuffer",
                TensorCopyFromBuffer
            ),
            tensor_copy_to_buffer: symbol!(
                tflite_library,
                "TfLiteTensorCopyToBuffer",
                TensorCopyToBuffer
            ),
            edgetpu_list_devices: symbol!(
                edgetpu_library,
                "edgetpu_list_devices",
                EdgeTpuListDevices
            ),
            edgetpu_free_devices: symbol!(
                edgetpu_library,
                "edgetpu_free_devices",
                EdgeTpuFreeDevices
            ),
            edgetpu_create_delegate: symbol!(
                edgetpu_library,
                "edgetpu_create_delegate",
                EdgeTpuCreateDelegate
            ),
            edgetpu_free_delegate: symbol!(
                edgetpu_library,
                "edgetpu_free_delegate",
                EdgeTpuFreeDelegate
            ),
            edgetpu_version: symbol!(edgetpu_library, "edgetpu_version", EdgeTpuVersion),
            _tflite_library: tflite_library,
            _edgetpu_library: edgetpu_library,
        };
        Ok(api)
    }
}
