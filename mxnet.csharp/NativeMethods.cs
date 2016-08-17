using System;
using System.Runtime.InteropServices;

namespace mxnet.csharp
{


    /// Return Type: void
    ///param0: char*
    ///param1: NDArrayHandle->void*
    ///param2: void*
    public delegate void ExecutorMonitorCallback([In] [MarshalAs(UnmanagedType.LPStr)] string param0, IntPtr param1, IntPtr param2);

    /// Return Type: void
    ///param0: int
    ///param1: float**
    ///param2: int*
    ///param3: unsigned int**
    ///param4: int*
    ///param5: void*
    public delegate void NativeOpInfo_forward(int param0, ref IntPtr param1, ref int param2, ref IntPtr param3, ref int param4, IntPtr param5);

    /// Return Type: void
    ///param0: int
    ///param1: float**
    ///param2: int*
    ///param3: unsigned int**
    ///param4: int*
    ///param5: void*
    public delegate void NativeOpInfo_backward(int param0, ref IntPtr param1, ref int param2, ref IntPtr param3, ref int param4, IntPtr param5);

    /// Return Type: void
    ///param0: int
    ///param1: int*
    ///param2: unsigned int**
    ///param3: void*
    public delegate void NativeOpInfo_infer_shape(int param0, ref int param1, ref IntPtr param2, IntPtr param3);

    /// Return Type: void
    ///param0: char***
    ///param1: void*
    public delegate void NativeOpInfo_list_outputs(ref IntPtr param0, IntPtr param1);

    /// Return Type: void
    ///param0: char***
    ///param1: void*
    public delegate void NativeOpInfo_list_arguments(ref IntPtr param0, IntPtr param1);

    [StructLayout(LayoutKind.Sequential)]
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
        public IntPtr p_forward;

        /// void*
        public IntPtr p_backward;

        /// void*
        public IntPtr p_infer_shape;

        /// void*
        public IntPtr p_list_outputs;

        /// void*
        public IntPtr p_list_arguments;
    }

    /// Return Type: boolean
    ///param0: int
    ///param1: void**
    ///param2: int*
    ///param3: void*
    public delegate bool NDArrayOpInfo_forward(int param0, ref IntPtr param1, ref int param2, IntPtr param3);

    /// Return Type: boolean
    ///param0: int
    ///param1: void**
    ///param2: int*
    ///param3: void*
    public delegate bool NDArrayOpInfo_backward(int param0, ref IntPtr param1, ref int param2, IntPtr param3);

    /// Return Type: boolean
    ///param0: int
    ///param1: int*
    ///param2: unsigned int**
    ///param3: void*
    public delegate bool NDArrayOpInfo_infer_shape(int param0, ref int param1, ref IntPtr param2, IntPtr param3);

    /// Return Type: boolean
    ///param0: char***
    ///param1: void*
    public delegate bool NDArrayOpInfo_list_outputs(ref IntPtr param0, IntPtr param1);

    /// Return Type: boolean
    ///param0: char***
    ///param1: void*
    public delegate bool NDArrayOpInfo_list_arguments(ref IntPtr param0, IntPtr param1);

    /// Return Type: boolean
    ///param0: int*
    ///param1: int*
    ///param2: int*
    ///param3: int*
    ///param4: int**
    ///param5: void*
    public delegate bool NDArrayOpInfo_declare_backward_dependency(ref int param0, ref int param1, ref int param2, ref int param3, ref IntPtr param4, IntPtr param5);

    [StructLayout(LayoutKind.Sequential)]
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
        public IntPtr p_forward;

        /// void*
        public IntPtr p_backward;

        /// void*
        public IntPtr p_infer_shape;

        /// void*
        public IntPtr p_list_outputs;

        /// void*
        public IntPtr p_list_arguments;

        /// void*
        public IntPtr p_declare_backward_dependency;
    }

    /// Return Type: boolean
    ///param0: int
    ///param1: void**
    ///param2: int*
    ///param3: int*
    ///param4: boolean
    ///param5: void*
    public delegate bool CustomOpInfo_forward(int param0, ref IntPtr param1, ref int param2, ref int param3, [MarshalAs(UnmanagedType.I1)] bool param4, IntPtr param5);

    /// Return Type: boolean
    ///param0: int
    ///param1: void**
    ///param2: int*
    ///param3: int*
    ///param4: boolean
    ///param5: void*
    public delegate bool CustomOpInfo_backward(int param0, ref IntPtr param1, ref int param2, ref int param3, [MarshalAs(UnmanagedType.I1)] bool param4, IntPtr param5);

    /// Return Type: boolean
    ///param0: void*
    public delegate bool CustomOpInfo_del(IntPtr param0);

    [StructLayout(LayoutKind.Sequential)]
    public struct CustomOpInfo
    {

        /// CustomOpInfo_forward
        public CustomOpInfo_forward AnonymousMember1;

        /// CustomOpInfo_backward
        public CustomOpInfo_backward AnonymousMember2;

        /// CustomOpInfo_del
        public CustomOpInfo_del AnonymousMember3;

        /// void*
        public IntPtr p_forward;

        /// void*
        public IntPtr p_backward;

        /// void*
        public IntPtr p_del;
    }

    /// Return Type: boolean
    ///param0: char***
    ///param1: void*
    public delegate bool CustomOpPropInfo_list_arguments(ref IntPtr param0, IntPtr param1);

    /// Return Type: boolean
    ///param0: char***
    ///param1: void*
    public delegate bool CustomOpPropInfo_list_outputs(ref IntPtr param0, IntPtr param1);

    /// Return Type: boolean
    ///param0: int
    ///param1: int*
    ///param2: unsigned int**
    ///param3: void*
    public delegate bool CustomOpPropInfo_infer_shape(int param0, ref int param1, ref IntPtr param2, IntPtr param3);

    /// Return Type: boolean
    ///param0: int*
    ///param1: int*
    ///param2: int*
    ///param3: int*
    ///param4: int**
    ///param5: void*
    public delegate bool CustomOpPropInfo_declare_backward_dependency(ref int param0, ref int param1, ref int param2, ref int param3, ref IntPtr param4, IntPtr param5);

    /// Return Type: boolean
    ///param0: char*
    ///param1: int
    ///param2: unsigned int**
    ///param3: int*
    ///param4: int*
    ///param5: CustomOpInfo*
    ///param6: void*
    public delegate bool CustomOpPropInfo_create_operator([In] [MarshalAs(UnmanagedType.LPStr)] string param0, int param1, ref IntPtr param2, ref int param3, ref int param4, ref CustomOpInfo param5, IntPtr param6);

    /// Return Type: boolean
    ///param0: char***
    ///param1: void*
    public delegate bool CustomOpPropInfo_list_auxiliary_states(ref IntPtr param0, IntPtr param1);

    /// Return Type: boolean
    ///param0: void*
    public delegate bool CustomOpPropInfo_del(IntPtr param0);

    [StructLayout(LayoutKind.Sequential)]
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
        public IntPtr p_list_arguments;

        /// void*
        public IntPtr p_list_outputs;

        /// void*
        public IntPtr p_infer_shape;

        /// void*
        public IntPtr p_declare_backward_dependency;

        /// void*
        public IntPtr p_create_operator;

        /// void*
        public IntPtr p_list_auxiliary_states;

        /// void*
        public IntPtr p_del;
    }

    /// Return Type: boolean
    ///param0: char*
    ///param1: int
    ///param2: char**
    ///param3: char**
    ///param4: CustomOpPropInfo*
    public delegate bool CustomOpPropCreator([In] [MarshalAs(UnmanagedType.LPStr)] string param0, int param1, ref IntPtr param2, ref IntPtr param3, ref CustomOpPropInfo param4);

    public class NativeMethods
    {

        /// Return Type: char*
        [DllImport("libmxnet.dll", EntryPoint = "MXGetLastError")]
        public static extern IntPtr MXGetLastError();


        /// Return Type: int
        ///seed: int
        [DllImport("libmxnet.dll", EntryPoint = "MXRandomSeed")]
        public static extern int MXRandomSeed(int seed);


        /// Return Type: int
        [DllImport("libmxnet.dll", EntryPoint = "MXNotifyShutdown")]
        public static extern int MXNotifyShutdown();


        /// Return Type: int
        ///out: NDArrayHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArrayCreateNone")]
        public static extern int MXNDArrayCreateNone(ref IntPtr @out);


        /// Return Type: int
        ///shape: mx_uint*
        ///ndim: mx_uint->unsigned int
        ///dev_type: int
        ///dev_id: int
        ///delay_alloc: int
        ///out: NDArrayHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArrayCreate")]
        public static extern int MXNDArrayCreate(ref uint shape, uint ndim, int dev_type, int dev_id, int delay_alloc, ref IntPtr @out);


        /// Return Type: int
        ///shape: mx_uint*
        ///ndim: mx_uint->unsigned int
        ///dev_type: int
        ///dev_id: int
        ///delay_alloc: int
        ///dtype: int
        ///out: NDArrayHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArrayCreateEx")]
        public static extern int MXNDArrayCreateEx(ref uint shape, uint ndim, int dev_type, int dev_id, int delay_alloc, int dtype, ref IntPtr @out);


        /// Return Type: int
        ///buf: void*
        ///size: size_t->unsigned int
        ///out: NDArrayHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArrayLoadFromRawBytes")]
        public static extern int MXNDArrayLoadFromRawBytes(IntPtr buf, [MarshalAs(UnmanagedType.SysUInt)] uint size, ref IntPtr @out);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///out_size: size_t*
        ///out_buf: char**
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArraySaveRawBytes")]
        public static extern int MXNDArraySaveRawBytes(IntPtr handle, ref uint out_size, ref IntPtr out_buf);


        /// Return Type: int
        ///fname: char*
        ///num_args: mx_uint->unsigned int
        ///args: NDArrayHandle*
        ///keys: char**
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArraySave")]
        public static extern int MXNDArraySave([In] [MarshalAs(UnmanagedType.LPStr)] string fname, uint num_args, ref IntPtr args, ref IntPtr keys);


        /// Return Type: int
        ///fname: char*
        ///out_size: mx_uint*
        ///out_arr: NDArrayHandle**
        ///out_name_size: mx_uint*
        ///out_names: char***
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArrayLoad")]
        public static extern int MXNDArrayLoad([In] [MarshalAs(UnmanagedType.LPStr)] string fname, ref uint out_size, ref IntPtr out_arr, ref uint out_name_size, ref IntPtr out_names);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///data: void*
        ///size: size_t->unsigned int
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArraySyncCopyFromCPU")]
        public static extern int MXNDArraySyncCopyFromCPU(IntPtr handle, IntPtr data, [MarshalAs(UnmanagedType.SysUInt)] uint size);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///data: void*
        ///size: size_t->unsigned int
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArraySyncCopyToCPU")]
        public static extern int MXNDArraySyncCopyToCPU(IntPtr handle, IntPtr data, [MarshalAs(UnmanagedType.SysUInt)] uint size);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArrayWaitToRead")]
        public static extern int MXNDArrayWaitToRead(IntPtr handle);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArrayWaitToWrite")]
        public static extern int MXNDArrayWaitToWrite(IntPtr handle);


        /// Return Type: int
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArrayWaitAll")]
        public static extern int MXNDArrayWaitAll();


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArrayFree")]
        public static extern int MXNDArrayFree(IntPtr handle);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///slice_begin: mx_uint->unsigned int
        ///slice_end: mx_uint->unsigned int
        ///out: NDArrayHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArraySlice")]
        public static extern int MXNDArraySlice(IntPtr handle, uint slice_begin, uint slice_end, ref IntPtr @out);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///idx: mx_uint->unsigned int
        ///out: NDArrayHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArrayAt")]
        public static extern int MXNDArrayAt(IntPtr handle, uint idx, ref IntPtr @out);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///ndim: int
        ///dims: int*
        ///out: NDArrayHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArrayReshape")]
        public static extern int MXNDArrayReshape(IntPtr handle, int ndim, ref int dims, ref IntPtr @out);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///out_dim: mx_uint*
        ///out_pdata: mx_uint**
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArrayGetShape")]
        public static extern int MXNDArrayGetShape(IntPtr handle, ref uint out_dim, ref IntPtr out_pdata);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///out_pdata: mx_float**
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArrayGetData")]
        public static extern int MXNDArrayGetData(IntPtr handle, ref IntPtr out_pdata);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///out_dtype: int*
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArrayGetDType")]
        public static extern int MXNDArrayGetDType(IntPtr handle, ref int out_dtype);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///out_dev_type: int*
        ///out_dev_id: int*
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArrayGetContext")]
        public static extern int MXNDArrayGetContext(IntPtr handle, ref int out_dev_type, ref int out_dev_id);


        /// Return Type: int
        ///out_size: mx_uint*
        ///out_array: FunctionHandle**
        [DllImport("libmxnet.dll", EntryPoint = "MXListFunctions")]
        public static extern int MXListFunctions(ref uint out_size, ref IntPtr out_array);


        /// Return Type: int
        ///name: char*
        ///out: FunctionHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXGetFunction")]
        public static extern int MXGetFunction([In] [MarshalAs(UnmanagedType.LPStr)] string name, ref IntPtr @out);


        /// Return Type: int
        ///fun: FunctionHandle->void*
        ///name: char**
        ///description: char**
        ///num_args: mx_uint*
        ///arg_names: char***
        ///arg_type_infos: char***
        ///arg_descriptions: char***
        ///return_type: char**
        [DllImport("libmxnet.dll", EntryPoint = "MXFuncGetInfo")]
        public static extern int MXFuncGetInfo(IntPtr fun, ref IntPtr name, ref IntPtr description, ref uint num_args, ref IntPtr arg_names, ref IntPtr arg_type_infos, ref IntPtr arg_descriptions, ref IntPtr return_type);


        /// Return Type: int
        ///fun: FunctionHandle->void*
        ///num_use_vars: mx_uint*
        ///num_scalars: mx_uint*
        ///num_mutate_vars: mx_uint*
        ///type_mask: int*
        [DllImport("libmxnet.dll", EntryPoint = "MXFuncDescribe")]
        public static extern int MXFuncDescribe(IntPtr fun, ref uint num_use_vars, ref uint num_scalars, ref uint num_mutate_vars, ref int type_mask);


        /// Return Type: int
        ///fun: FunctionHandle->void*
        ///use_vars: NDArrayHandle*
        ///scalar_args: mx_float*
        ///mutate_vars: NDArrayHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXFuncInvoke")]
        public static extern int MXFuncInvoke(IntPtr fun, ref IntPtr use_vars, ref float scalar_args, ref IntPtr mutate_vars);


        /// Return Type: int
        ///fun: FunctionHandle->void*
        ///use_vars: NDArrayHandle*
        ///scalar_args: mx_float*
        ///mutate_vars: NDArrayHandle*
        ///num_params: int
        ///param_keys: char**
        ///param_vals: char**
        [DllImport("libmxnet.dll", EntryPoint = "MXFuncInvokeEx")]
        public static extern int MXFuncInvokeEx(IntPtr fun, ref IntPtr use_vars, ref float scalar_args, ref IntPtr mutate_vars, int num_params, ref IntPtr param_keys, ref IntPtr param_vals);


        /// Return Type: int
        ///out_size: mx_uint*
        ///out_array: AtomicSymbolCreator**
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolListAtomicSymbolCreators")]
        public static extern int MXSymbolListAtomicSymbolCreators(out uint out_size,
            out IntPtr out_array_ptr);


        /// Return Type: int
        ///creator: AtomicSymbolCreator->void*
        ///name: char**
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolGetAtomicSymbolName")]
        public static extern int MXSymbolGetAtomicSymbolName(IntPtr creator, ref IntPtr name);


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
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolGetAtomicSymbolInfo")]
        public static extern int MXSymbolGetAtomicSymbolInfo(IntPtr creator,
          [Out]out IntPtr name,
          [Out]out IntPtr description,
          [Out]out uint num_args,
          [Out]out IntPtr arg_names,
          [Out]out IntPtr arg_type_infos,
          [Out]out IntPtr arg_descriptions,
          [Out]out IntPtr key_var_num_args,
          [Out]out IntPtr return_type);


        /// Return Type: int
        ///creator: AtomicSymbolCreator->void*
        ///num_param: mx_uint->unsigned int
        ///keys: char**
        ///vals: char**
        ///out: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolCreateAtomicSymbol")]
        public static extern int MXSymbolCreateAtomicSymbol(IntPtr creator, 
            uint num_param,
         [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] keys ,
         [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] vals, 
            out IntPtr @out);


        /// Return Type: int
        ///name: char*
        ///out: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolCreateVariable")]
        public static extern int MXSymbolCreateVariable([In] [MarshalAs(UnmanagedType.LPStr)] string name, out IntPtr @out);


        /// Return Type: int
        ///num_symbols: mx_uint->unsigned int
        ///symbols: SymbolHandle*
        ///out: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolCreateGroup")]
        public static extern int MXSymbolCreateGroup(uint num_symbols, ref IntPtr symbols, ref IntPtr @out);


        /// Return Type: int
        ///fname: char*
        ///out: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolCreateFromFile")]
        public static extern int MXSymbolCreateFromFile([In] [MarshalAs(UnmanagedType.LPStr)] string fname, ref IntPtr @out);


        /// Return Type: int
        ///json: char*
        ///out: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolCreateFromJSON")]
        public static extern int MXSymbolCreateFromJSON([In] [MarshalAs(UnmanagedType.LPStr)] string json, ref IntPtr @out);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///fname: char*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolSaveToFile")]
        public static extern int MXSymbolSaveToFile(IntPtr symbol, [In] [MarshalAs(UnmanagedType.LPStr)] string fname);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out_json: char**
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolSaveToJSON")]
        public static extern int MXSymbolSaveToJSON(IntPtr symbol, ref IntPtr out_json);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolFree")]
        public static extern int MXSymbolFree(IntPtr symbol);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolCopy")]
        public static extern int MXSymbolCopy(IntPtr symbol, ref IntPtr @out);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out_str: char**
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolPrint")]
        public static extern int MXSymbolPrint(IntPtr symbol, ref IntPtr out_str);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out: char**
        ///success: int*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolGetName")]
        public static extern int MXSymbolGetName(IntPtr symbol, ref IntPtr @out, ref int success);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///key: char*
        ///out: char**
        ///success: int*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolGetAttr")]
        public static extern int MXSymbolGetAttr(IntPtr symbol, [In] [MarshalAs(UnmanagedType.LPStr)] string key, ref IntPtr @out, ref int success);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///key: char*
        ///value: char*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolSetAttr")]
        public static extern int MXSymbolSetAttr(IntPtr symbol, [In] [MarshalAs(UnmanagedType.LPStr)] string key, [In] [MarshalAs(UnmanagedType.LPStr)] string value);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out_size: mx_uint*
        ///out: char***
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolListAttr")]
        public static extern int MXSymbolListAttr(IntPtr symbol, ref uint out_size, ref IntPtr @out);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out_size: mx_uint*
        ///out: char***
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolListAttrShallow")]
        public static extern int MXSymbolListAttrShallow(IntPtr symbol, ref uint out_size, ref IntPtr @out);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out_size: mx_uint*
        ///out_str_array: char***
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolListArguments")]
        public static extern int MXSymbolListArguments(IntPtr symbol, ref uint out_size, ref IntPtr out_str_array);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out_size: mx_uint*
        ///out_str_array: char***
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolListOutputs")]
        public static extern int MXSymbolListOutputs(IntPtr symbol, ref uint out_size, ref IntPtr out_str_array);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolGetInternals")]
        public static extern int MXSymbolGetInternals(IntPtr symbol, ref IntPtr @out);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///index: mx_uint->unsigned int
        ///out: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolGetOutput")]
        public static extern int MXSymbolGetOutput(IntPtr symbol, uint index, ref IntPtr @out);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out_size: mx_uint*
        ///out_str_array: char***
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolListAuxiliaryStates")]
        public static extern int MXSymbolListAuxiliaryStates(IntPtr symbol, ref uint out_size, ref IntPtr out_str_array);


        /// Return Type: int
        ///sym: SymbolHandle->void*
        ///name: char*
        ///num_args: mx_uint->unsigned int
        ///keys: char**
        ///args: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolCompose")]
        public static extern int MXSymbolCompose(IntPtr sym, 
            [In] [MarshalAs(UnmanagedType.LPStr)] string name,
            uint num_args,
           [In]  [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] keys,
           [In]  [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.SysInt)] IntPtr[] args);

        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolCompose")]
        public static extern int MXSymbolCompose(IntPtr sym,
          [In] [MarshalAs(UnmanagedType.LPStr)] string name,
          uint num_args,
         [In]  IntPtr keys,
         [In]  [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.SysInt)] IntPtr[] args);


        /// Return Type: int
        ///sym: SymbolHandle->void*
        ///num_wrt: mx_uint->unsigned int
        ///wrt: char**
        ///out: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolGrad")]
        public static extern int MXSymbolGrad(IntPtr sym, uint num_wrt, ref IntPtr wrt, ref IntPtr @out);


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
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolInferShape")]
        public static extern int MXSymbolInferShape(IntPtr sym, uint num_args, ref IntPtr keys, ref uint arg_ind_ptr, ref uint arg_shape_data, ref uint in_shape_size, ref IntPtr in_shape_ndim, ref IntPtr in_shape_data, ref uint out_shape_size, ref IntPtr out_shape_ndim, ref IntPtr out_shape_data, ref uint aux_shape_size, ref IntPtr aux_shape_ndim, ref IntPtr aux_shape_data, ref int complete);

    }

}
