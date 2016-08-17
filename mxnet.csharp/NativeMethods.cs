using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.csharp
{


    /// Return Type: void
    ///param0: char*
    ///param1: NDArrayHandle->void*
    ///param2: void*
    public delegate void ExecutorMonitorCallback([System.Runtime.InteropServices.InAttribute()] [System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.LPStr)] string param0, System.IntPtr param1, System.IntPtr param2);

    /// Return Type: void
    ///param0: int
    ///param1: float**
    ///param2: int*
    ///param3: unsigned int**
    ///param4: int*
    ///param5: void*
    public delegate void NativeOpInfo_forward(int param0, ref System.IntPtr param1, ref int param2, ref System.IntPtr param3, ref int param4, System.IntPtr param5);

    /// Return Type: void
    ///param0: int
    ///param1: float**
    ///param2: int*
    ///param3: unsigned int**
    ///param4: int*
    ///param5: void*
    public delegate void NativeOpInfo_backward(int param0, ref System.IntPtr param1, ref int param2, ref System.IntPtr param3, ref int param4, System.IntPtr param5);

    /// Return Type: void
    ///param0: int
    ///param1: int*
    ///param2: unsigned int**
    ///param3: void*
    public delegate void NativeOpInfo_infer_shape(int param0, ref int param1, ref System.IntPtr param2, System.IntPtr param3);

    /// Return Type: void
    ///param0: char***
    ///param1: void*
    public delegate void NativeOpInfo_list_outputs(ref System.IntPtr param0, System.IntPtr param1);

    /// Return Type: void
    ///param0: char***
    ///param1: void*
    public delegate void NativeOpInfo_list_arguments(ref System.IntPtr param0, System.IntPtr param1);

    [System.Runtime.InteropServices.StructLayoutAttribute(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct NativeOpInfo
    {

        /// NativeOpInfo_forward
        public NativeOpInfo_forward AnonymousMember1;

        /// NativeOpInfo_backward
        public NativeOpInfo_backward AnonymousMember2;

        /// NativeOpInfo_infer_shape
        public NativeOpInfo_infer_shape AnonymousMember3;

        /// NativeOpInfo_list_outputs
        public NativeOpInfo_list_outputs AnonymousMember4;

        /// NativeOpInfo_list_arguments
        public NativeOpInfo_list_arguments AnonymousMember5;

        /// void*
        public System.IntPtr p_forward;

        /// void*
        public System.IntPtr p_backward;

        /// void*
        public System.IntPtr p_infer_shape;

        /// void*
        public System.IntPtr p_list_outputs;

        /// void*
        public System.IntPtr p_list_arguments;
    }

    /// Return Type: boolean
    ///param0: int
    ///param1: void**
    ///param2: int*
    ///param3: void*
    public delegate bool NDArrayOpInfo_forward(int param0, ref System.IntPtr param1, ref int param2, System.IntPtr param3);

    /// Return Type: boolean
    ///param0: int
    ///param1: void**
    ///param2: int*
    ///param3: void*
    public delegate bool NDArrayOpInfo_backward(int param0, ref System.IntPtr param1, ref int param2, System.IntPtr param3);

    /// Return Type: boolean
    ///param0: int
    ///param1: int*
    ///param2: unsigned int**
    ///param3: void*
    public delegate bool NDArrayOpInfo_infer_shape(int param0, ref int param1, ref System.IntPtr param2, System.IntPtr param3);

    /// Return Type: boolean
    ///param0: char***
    ///param1: void*
    public delegate bool NDArrayOpInfo_list_outputs(ref System.IntPtr param0, System.IntPtr param1);

    /// Return Type: boolean
    ///param0: char***
    ///param1: void*
    public delegate bool NDArrayOpInfo_list_arguments(ref System.IntPtr param0, System.IntPtr param1);

    /// Return Type: boolean
    ///param0: int*
    ///param1: int*
    ///param2: int*
    ///param3: int*
    ///param4: int**
    ///param5: void*
    public delegate bool NDArrayOpInfo_declare_backward_dependency(ref int param0, ref int param1, ref int param2, ref int param3, ref System.IntPtr param4, System.IntPtr param5);

    [System.Runtime.InteropServices.StructLayoutAttribute(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct NDArrayOpInfo
    {

        /// NDArrayOpInfo_forward
        public NDArrayOpInfo_forward AnonymousMember1;

        /// NDArrayOpInfo_backward
        public NDArrayOpInfo_backward AnonymousMember2;

        /// NDArrayOpInfo_infer_shape
        public NDArrayOpInfo_infer_shape AnonymousMember3;

        /// NDArrayOpInfo_list_outputs
        public NDArrayOpInfo_list_outputs AnonymousMember4;

        /// NDArrayOpInfo_list_arguments
        public NDArrayOpInfo_list_arguments AnonymousMember5;

        /// NDArrayOpInfo_declare_backward_dependency
        public NDArrayOpInfo_declare_backward_dependency AnonymousMember6;

        /// void*
        public System.IntPtr p_forward;

        /// void*
        public System.IntPtr p_backward;

        /// void*
        public System.IntPtr p_infer_shape;

        /// void*
        public System.IntPtr p_list_outputs;

        /// void*
        public System.IntPtr p_list_arguments;

        /// void*
        public System.IntPtr p_declare_backward_dependency;
    }

    /// Return Type: boolean
    ///param0: int
    ///param1: void**
    ///param2: int*
    ///param3: int*
    ///param4: boolean
    ///param5: void*
    public delegate bool CustomOpInfo_forward(int param0, ref System.IntPtr param1, ref int param2, ref int param3, [System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.I1)] bool param4, System.IntPtr param5);

    /// Return Type: boolean
    ///param0: int
    ///param1: void**
    ///param2: int*
    ///param3: int*
    ///param4: boolean
    ///param5: void*
    public delegate bool CustomOpInfo_backward(int param0, ref System.IntPtr param1, ref int param2, ref int param3, [System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.I1)] bool param4, System.IntPtr param5);

    /// Return Type: boolean
    ///param0: void*
    public delegate bool CustomOpInfo_del(System.IntPtr param0);

    [System.Runtime.InteropServices.StructLayoutAttribute(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct CustomOpInfo
    {

        /// CustomOpInfo_forward
        public CustomOpInfo_forward AnonymousMember1;

        /// CustomOpInfo_backward
        public CustomOpInfo_backward AnonymousMember2;

        /// CustomOpInfo_del
        public CustomOpInfo_del AnonymousMember3;

        /// void*
        public System.IntPtr p_forward;

        /// void*
        public System.IntPtr p_backward;

        /// void*
        public System.IntPtr p_del;
    }

    /// Return Type: boolean
    ///param0: char***
    ///param1: void*
    public delegate bool CustomOpPropInfo_list_arguments(ref System.IntPtr param0, System.IntPtr param1);

    /// Return Type: boolean
    ///param0: char***
    ///param1: void*
    public delegate bool CustomOpPropInfo_list_outputs(ref System.IntPtr param0, System.IntPtr param1);

    /// Return Type: boolean
    ///param0: int
    ///param1: int*
    ///param2: unsigned int**
    ///param3: void*
    public delegate bool CustomOpPropInfo_infer_shape(int param0, ref int param1, ref System.IntPtr param2, System.IntPtr param3);

    /// Return Type: boolean
    ///param0: int*
    ///param1: int*
    ///param2: int*
    ///param3: int*
    ///param4: int**
    ///param5: void*
    public delegate bool CustomOpPropInfo_declare_backward_dependency(ref int param0, ref int param1, ref int param2, ref int param3, ref System.IntPtr param4, System.IntPtr param5);

    /// Return Type: boolean
    ///param0: char*
    ///param1: int
    ///param2: unsigned int**
    ///param3: int*
    ///param4: int*
    ///param5: CustomOpInfo*
    ///param6: void*
    public delegate bool CustomOpPropInfo_create_operator([System.Runtime.InteropServices.InAttribute()] [System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.LPStr)] string param0, int param1, ref System.IntPtr param2, ref int param3, ref int param4, ref CustomOpInfo param5, System.IntPtr param6);

    /// Return Type: boolean
    ///param0: char***
    ///param1: void*
    public delegate bool CustomOpPropInfo_list_auxiliary_states(ref System.IntPtr param0, System.IntPtr param1);

    /// Return Type: boolean
    ///param0: void*
    public delegate bool CustomOpPropInfo_del(System.IntPtr param0);

    [System.Runtime.InteropServices.StructLayoutAttribute(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct CustomOpPropInfo
    {

        /// CustomOpPropInfo_list_arguments
        public CustomOpPropInfo_list_arguments AnonymousMember1;

        /// CustomOpPropInfo_list_outputs
        public CustomOpPropInfo_list_outputs AnonymousMember2;

        /// CustomOpPropInfo_infer_shape
        public CustomOpPropInfo_infer_shape AnonymousMember3;

        /// CustomOpPropInfo_declare_backward_dependency
        public CustomOpPropInfo_declare_backward_dependency AnonymousMember4;

        /// CustomOpPropInfo_create_operator
        public CustomOpPropInfo_create_operator AnonymousMember5;

        /// CustomOpPropInfo_list_auxiliary_states
        public CustomOpPropInfo_list_auxiliary_states AnonymousMember6;

        /// CustomOpPropInfo_del
        public CustomOpPropInfo_del AnonymousMember7;

        /// void*
        public System.IntPtr p_list_arguments;

        /// void*
        public System.IntPtr p_list_outputs;

        /// void*
        public System.IntPtr p_infer_shape;

        /// void*
        public System.IntPtr p_declare_backward_dependency;

        /// void*
        public System.IntPtr p_create_operator;

        /// void*
        public System.IntPtr p_list_auxiliary_states;

        /// void*
        public System.IntPtr p_del;
    }

    /// Return Type: boolean
    ///param0: char*
    ///param1: int
    ///param2: char**
    ///param3: char**
    ///param4: CustomOpPropInfo*
    public delegate bool CustomOpPropCreator([System.Runtime.InteropServices.InAttribute()] [System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.LPStr)] string param0, int param1, ref System.IntPtr param2, ref System.IntPtr param3, ref CustomOpPropInfo param4);

    public partial class NativeMethods
    {

        /// Return Type: char*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXGetLastError")]
        public static extern System.IntPtr MXGetLastError();


        /// Return Type: int
        ///seed: int
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXRandomSeed")]
        public static extern int MXRandomSeed(int seed);


        /// Return Type: int
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXNotifyShutdown")]
        public static extern int MXNotifyShutdown();


        /// Return Type: int
        ///out: NDArrayHandle*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXNDArrayCreateNone")]
        public static extern int MXNDArrayCreateNone(ref System.IntPtr @out);


        /// Return Type: int
        ///shape: mx_uint*
        ///ndim: mx_uint->unsigned int
        ///dev_type: int
        ///dev_id: int
        ///delay_alloc: int
        ///out: NDArrayHandle*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXNDArrayCreate")]
        public static extern int MXNDArrayCreate(ref uint shape, uint ndim, int dev_type, int dev_id, int delay_alloc, ref System.IntPtr @out);


        /// Return Type: int
        ///shape: mx_uint*
        ///ndim: mx_uint->unsigned int
        ///dev_type: int
        ///dev_id: int
        ///delay_alloc: int
        ///dtype: int
        ///out: NDArrayHandle*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXNDArrayCreateEx")]
        public static extern int MXNDArrayCreateEx(ref uint shape, uint ndim, int dev_type, int dev_id, int delay_alloc, int dtype, ref System.IntPtr @out);


        /// Return Type: int
        ///buf: void*
        ///size: size_t->unsigned int
        ///out: NDArrayHandle*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXNDArrayLoadFromRawBytes")]
        public static extern int MXNDArrayLoadFromRawBytes(System.IntPtr buf, [System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.SysUInt)] uint size, ref System.IntPtr @out);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///out_size: size_t*
        ///out_buf: char**
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXNDArraySaveRawBytes")]
        public static extern int MXNDArraySaveRawBytes(System.IntPtr handle, ref uint out_size, ref System.IntPtr out_buf);


        /// Return Type: int
        ///fname: char*
        ///num_args: mx_uint->unsigned int
        ///args: NDArrayHandle*
        ///keys: char**
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXNDArraySave")]
        public static extern int MXNDArraySave([System.Runtime.InteropServices.InAttribute()] [System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.LPStr)] string fname, uint num_args, ref System.IntPtr args, ref System.IntPtr keys);


        /// Return Type: int
        ///fname: char*
        ///out_size: mx_uint*
        ///out_arr: NDArrayHandle**
        ///out_name_size: mx_uint*
        ///out_names: char***
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXNDArrayLoad")]
        public static extern int MXNDArrayLoad([System.Runtime.InteropServices.InAttribute()] [System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.LPStr)] string fname, ref uint out_size, ref System.IntPtr out_arr, ref uint out_name_size, ref System.IntPtr out_names);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///data: void*
        ///size: size_t->unsigned int
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXNDArraySyncCopyFromCPU")]
        public static extern int MXNDArraySyncCopyFromCPU(System.IntPtr handle, System.IntPtr data, [System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.SysUInt)] uint size);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///data: void*
        ///size: size_t->unsigned int
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXNDArraySyncCopyToCPU")]
        public static extern int MXNDArraySyncCopyToCPU(System.IntPtr handle, System.IntPtr data, [System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.SysUInt)] uint size);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXNDArrayWaitToRead")]
        public static extern int MXNDArrayWaitToRead(System.IntPtr handle);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXNDArrayWaitToWrite")]
        public static extern int MXNDArrayWaitToWrite(System.IntPtr handle);


        /// Return Type: int
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXNDArrayWaitAll")]
        public static extern int MXNDArrayWaitAll();


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXNDArrayFree")]
        public static extern int MXNDArrayFree(System.IntPtr handle);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///slice_begin: mx_uint->unsigned int
        ///slice_end: mx_uint->unsigned int
        ///out: NDArrayHandle*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXNDArraySlice")]
        public static extern int MXNDArraySlice(System.IntPtr handle, uint slice_begin, uint slice_end, ref System.IntPtr @out);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///idx: mx_uint->unsigned int
        ///out: NDArrayHandle*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXNDArrayAt")]
        public static extern int MXNDArrayAt(System.IntPtr handle, uint idx, ref System.IntPtr @out);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///ndim: int
        ///dims: int*
        ///out: NDArrayHandle*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXNDArrayReshape")]
        public static extern int MXNDArrayReshape(System.IntPtr handle, int ndim, ref int dims, ref System.IntPtr @out);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///out_dim: mx_uint*
        ///out_pdata: mx_uint**
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXNDArrayGetShape")]
        public static extern int MXNDArrayGetShape(System.IntPtr handle, ref uint out_dim, ref System.IntPtr out_pdata);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///out_pdata: mx_float**
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXNDArrayGetData")]
        public static extern int MXNDArrayGetData(System.IntPtr handle, ref System.IntPtr out_pdata);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///out_dtype: int*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXNDArrayGetDType")]
        public static extern int MXNDArrayGetDType(System.IntPtr handle, ref int out_dtype);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///out_dev_type: int*
        ///out_dev_id: int*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXNDArrayGetContext")]
        public static extern int MXNDArrayGetContext(System.IntPtr handle, ref int out_dev_type, ref int out_dev_id);


        /// Return Type: int
        ///out_size: mx_uint*
        ///out_array: FunctionHandle**
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXListFunctions")]
        public static extern int MXListFunctions(ref uint out_size, ref System.IntPtr out_array);


        /// Return Type: int
        ///name: char*
        ///out: FunctionHandle*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXGetFunction")]
        public static extern int MXGetFunction([System.Runtime.InteropServices.InAttribute()] [System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.LPStr)] string name, ref System.IntPtr @out);


        /// Return Type: int
        ///fun: FunctionHandle->void*
        ///name: char**
        ///description: char**
        ///num_args: mx_uint*
        ///arg_names: char***
        ///arg_type_infos: char***
        ///arg_descriptions: char***
        ///return_type: char**
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXFuncGetInfo")]
        public static extern int MXFuncGetInfo(System.IntPtr fun, ref System.IntPtr name, ref System.IntPtr description, ref uint num_args, ref System.IntPtr arg_names, ref System.IntPtr arg_type_infos, ref System.IntPtr arg_descriptions, ref System.IntPtr return_type);


        /// Return Type: int
        ///fun: FunctionHandle->void*
        ///num_use_vars: mx_uint*
        ///num_scalars: mx_uint*
        ///num_mutate_vars: mx_uint*
        ///type_mask: int*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXFuncDescribe")]
        public static extern int MXFuncDescribe(System.IntPtr fun, ref uint num_use_vars, ref uint num_scalars, ref uint num_mutate_vars, ref int type_mask);


        /// Return Type: int
        ///fun: FunctionHandle->void*
        ///use_vars: NDArrayHandle*
        ///scalar_args: mx_float*
        ///mutate_vars: NDArrayHandle*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXFuncInvoke")]
        public static extern int MXFuncInvoke(System.IntPtr fun, ref System.IntPtr use_vars, ref float scalar_args, ref System.IntPtr mutate_vars);


        /// Return Type: int
        ///fun: FunctionHandle->void*
        ///use_vars: NDArrayHandle*
        ///scalar_args: mx_float*
        ///mutate_vars: NDArrayHandle*
        ///num_params: int
        ///param_keys: char**
        ///param_vals: char**
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXFuncInvokeEx")]
        public static extern int MXFuncInvokeEx(System.IntPtr fun, ref System.IntPtr use_vars, ref float scalar_args, ref System.IntPtr mutate_vars, int num_params, ref System.IntPtr param_keys, ref System.IntPtr param_vals);


        /// Return Type: int
        ///out_size: mx_uint*
        ///out_array: AtomicSymbolCreator**
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolListAtomicSymbolCreators")]
        public static extern int MXSymbolListAtomicSymbolCreators(ref uint out_size, ref System.IntPtr out_array);


        /// Return Type: int
        ///creator: AtomicSymbolCreator->void*
        ///name: char**
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolGetAtomicSymbolName")]
        public static extern int MXSymbolGetAtomicSymbolName(System.IntPtr creator, ref System.IntPtr name);


        /// Return Type: int
        ///creator: AtomicSymbolCreator->void*
        ///name: char**
        ///description: char**
        ///num_args: mx_uint*
        ///arg_names: char***
        ///arg_type_infos: char***
        ///arg_descriptions: char***
        ///key_var_num_args: char**
        ///return_type: char**
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolGetAtomicSymbolInfo")]
        public static extern int MXSymbolGetAtomicSymbolInfo(System.IntPtr creator, ref System.IntPtr name, ref System.IntPtr description, ref uint num_args, ref System.IntPtr arg_names, ref System.IntPtr arg_type_infos, ref System.IntPtr arg_descriptions, ref System.IntPtr key_var_num_args, ref System.IntPtr return_type);


        /// Return Type: int
        ///creator: AtomicSymbolCreator->void*
        ///num_param: mx_uint->unsigned int
        ///keys: char**
        ///vals: char**
        ///out: SymbolHandle*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolCreateAtomicSymbol")]
        public static extern int MXSymbolCreateAtomicSymbol(System.IntPtr creator, uint num_param, ref System.IntPtr keys, ref System.IntPtr vals, ref System.IntPtr @out);


        /// Return Type: int
        ///name: char*
        ///out: SymbolHandle*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolCreateVariable")]
        public static extern int MXSymbolCreateVariable([System.Runtime.InteropServices.InAttribute()] [System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.LPStr)] string name, ref System.IntPtr @out);


        /// Return Type: int
        ///num_symbols: mx_uint->unsigned int
        ///symbols: SymbolHandle*
        ///out: SymbolHandle*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolCreateGroup")]
        public static extern int MXSymbolCreateGroup(uint num_symbols, ref System.IntPtr symbols, ref System.IntPtr @out);


        /// Return Type: int
        ///fname: char*
        ///out: SymbolHandle*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolCreateFromFile")]
        public static extern int MXSymbolCreateFromFile([System.Runtime.InteropServices.InAttribute()] [System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.LPStr)] string fname, ref System.IntPtr @out);


        /// Return Type: int
        ///json: char*
        ///out: SymbolHandle*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolCreateFromJSON")]
        public static extern int MXSymbolCreateFromJSON([System.Runtime.InteropServices.InAttribute()] [System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.LPStr)] string json, ref System.IntPtr @out);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///fname: char*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolSaveToFile")]
        public static extern int MXSymbolSaveToFile(System.IntPtr symbol, [System.Runtime.InteropServices.InAttribute()] [System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.LPStr)] string fname);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out_json: char**
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolSaveToJSON")]
        public static extern int MXSymbolSaveToJSON(System.IntPtr symbol, ref System.IntPtr out_json);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolFree")]
        public static extern int MXSymbolFree(System.IntPtr symbol);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out: SymbolHandle*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolCopy")]
        public static extern int MXSymbolCopy(System.IntPtr symbol, ref System.IntPtr @out);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out_str: char**
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolPrint")]
        public static extern int MXSymbolPrint(System.IntPtr symbol, ref System.IntPtr out_str);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out: char**
        ///success: int*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolGetName")]
        public static extern int MXSymbolGetName(System.IntPtr symbol, ref System.IntPtr @out, ref int success);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///key: char*
        ///out: char**
        ///success: int*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolGetAttr")]
        public static extern int MXSymbolGetAttr(System.IntPtr symbol, [System.Runtime.InteropServices.InAttribute()] [System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.LPStr)] string key, ref System.IntPtr @out, ref int success);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///key: char*
        ///value: char*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolSetAttr")]
        public static extern int MXSymbolSetAttr(System.IntPtr symbol, [System.Runtime.InteropServices.InAttribute()] [System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.LPStr)] string key, [System.Runtime.InteropServices.InAttribute()] [System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.LPStr)] string value);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out_size: mx_uint*
        ///out: char***
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolListAttr")]
        public static extern int MXSymbolListAttr(System.IntPtr symbol, ref uint out_size, ref System.IntPtr @out);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out_size: mx_uint*
        ///out: char***
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolListAttrShallow")]
        public static extern int MXSymbolListAttrShallow(System.IntPtr symbol, ref uint out_size, ref System.IntPtr @out);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out_size: mx_uint*
        ///out_str_array: char***
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolListArguments")]
        public static extern int MXSymbolListArguments(System.IntPtr symbol, ref uint out_size, ref System.IntPtr out_str_array);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out_size: mx_uint*
        ///out_str_array: char***
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolListOutputs")]
        public static extern int MXSymbolListOutputs(System.IntPtr symbol, ref uint out_size, ref System.IntPtr out_str_array);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out: SymbolHandle*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolGetInternals")]
        public static extern int MXSymbolGetInternals(System.IntPtr symbol, ref System.IntPtr @out);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///index: mx_uint->unsigned int
        ///out: SymbolHandle*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolGetOutput")]
        public static extern int MXSymbolGetOutput(System.IntPtr symbol, uint index, ref System.IntPtr @out);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out_size: mx_uint*
        ///out_str_array: char***
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolListAuxiliaryStates")]
        public static extern int MXSymbolListAuxiliaryStates(System.IntPtr symbol, ref uint out_size, ref System.IntPtr out_str_array);


        /// Return Type: int
        ///sym: SymbolHandle->void*
        ///name: char*
        ///num_args: mx_uint->unsigned int
        ///keys: char**
        ///args: SymbolHandle*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolCompose")]
        public static extern int MXSymbolCompose(System.IntPtr sym, [System.Runtime.InteropServices.InAttribute()] [System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.LPStr)] string name, uint num_args, ref System.IntPtr keys, ref System.IntPtr args);


        /// Return Type: int
        ///sym: SymbolHandle->void*
        ///num_wrt: mx_uint->unsigned int
        ///wrt: char**
        ///out: SymbolHandle*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolGrad")]
        public static extern int MXSymbolGrad(System.IntPtr sym, uint num_wrt, ref System.IntPtr wrt, ref System.IntPtr @out);


        /// Return Type: int
        ///sym: SymbolHandle->void*
        ///num_args: mx_uint->unsigned int
        ///keys: char**
        ///arg_ind_ptr: mx_uint*
        ///arg_shape_data: mx_uint*
        ///in_shape_size: mx_uint*
        ///in_shape_ndim: mx_uint**
        ///in_shape_data: mx_uint***
        ///out_shape_size: mx_uint*
        ///out_shape_ndim: mx_uint**
        ///out_shape_data: mx_uint***
        ///aux_shape_size: mx_uint*
        ///aux_shape_ndim: mx_uint**
        ///aux_shape_data: mx_uint***
        ///complete: int*
        [System.Runtime.InteropServices.DllImportAttribute("libmxnet.dll", EntryPoint = "MXSymbolInferShape")]
        public static extern int MXSymbolInferShape(System.IntPtr sym, uint num_args, ref System.IntPtr keys, ref uint arg_ind_ptr, ref uint arg_shape_data, ref uint in_shape_size, ref System.IntPtr in_shape_ndim, ref System.IntPtr in_shape_data, ref uint out_shape_size, ref System.IntPtr out_shape_ndim, ref System.IntPtr out_shape_data, ref uint aux_shape_size, ref System.IntPtr aux_shape_ndim, ref System.IntPtr aux_shape_data, ref int complete);

    }

}
