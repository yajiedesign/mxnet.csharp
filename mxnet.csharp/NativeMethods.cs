using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Linq;
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
    public delegate void NativeOpInfoForward(int param0, ref IntPtr param1, ref int param2, ref IntPtr param3, ref int param4, IntPtr param5);

    /// Return Type: void
    ///param0: int
    ///param1: float**
    ///param2: int*
    ///param3: unsigned int**
    ///param4: int*
    ///param5: void*
    public delegate void NativeOpInfoBackward(int param0, ref IntPtr param1, ref int param2, ref IntPtr param3, ref int param4, IntPtr param5);

    /// Return Type: void
    ///param0: int
    ///param1: int*
    ///param2: unsigned int**
    ///param3: void*
    public delegate void NativeOpInfoInferShape(int param0, ref int param1, ref IntPtr param2, IntPtr param3);

    /// Return Type: void
    ///param0: char***
    ///param1: void*
    public delegate void NativeOpInfoListOutputs(ref IntPtr param0, IntPtr param1);

    /// Return Type: void
    ///param0: char***
    ///param1: void*
    public delegate void NativeOpInfoListArguments(ref IntPtr param0, IntPtr param1);

    [StructLayout(LayoutKind.Sequential)]
    public struct NativeOpInfo
    {

        /// NativeOpInfo_forward
        public NativeOpInfoForward AnonymousMember1;

        /// NativeOpInfo_backward
        public NativeOpInfoBackward AnonymousMember2;

        /// NativeOpInfo_infer_shape
        public NativeOpInfoInferShape AnonymousMember3;

        /// NativeOpInfo_list_outputs
        public NativeOpInfoListOutputs AnonymousMember4;

        /// NativeOpInfo_list_arguments
        public NativeOpInfoListArguments AnonymousMember5;

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
    public delegate bool NdArrayOpInfoForward(int param0, ref IntPtr param1, ref int param2, IntPtr param3);

    /// Return Type: boolean
    ///param0: int
    ///param1: void**
    ///param2: int*
    ///param3: void*
    public delegate bool NdArrayOpInfoBackward(int param0, ref IntPtr param1, ref int param2, IntPtr param3);

    /// Return Type: boolean
    ///param0: int
    ///param1: int*
    ///param2: unsigned int**
    ///param3: void*
    public delegate bool NdArrayOpInfoInferShape(int param0, ref int param1, ref IntPtr param2, IntPtr param3);

    /// Return Type: boolean
    ///param0: char***
    ///param1: void*
    public delegate bool NdArrayOpInfoListOutputs(ref IntPtr param0, IntPtr param1);

    /// Return Type: boolean
    ///param0: char***
    ///param1: void*
    public delegate bool NdArrayOpInfoListArguments(ref IntPtr param0, IntPtr param1);

    /// Return Type: boolean
    ///param0: int*
    ///param1: int*
    ///param2: int*
    ///param3: int*
    ///param4: int**
    ///param5: void*
    public delegate bool NdArrayOpInfoDeclareBackwardDependency(ref int param0, ref int param1, ref int param2, ref int param3, ref IntPtr param4, IntPtr param5);

    [StructLayout(LayoutKind.Sequential)]
    public struct NdArrayOpInfo
    {

        /// NDArrayOpInfo_forward
        public NdArrayOpInfoForward AnonymousMember1;

        /// NDArrayOpInfo_backward
        public NdArrayOpInfoBackward AnonymousMember2;

        /// NDArrayOpInfo_infer_shape
        public NdArrayOpInfoInferShape AnonymousMember3;

        /// NDArrayOpInfo_list_outputs
        public NdArrayOpInfoListOutputs AnonymousMember4;

        /// NDArrayOpInfo_list_arguments
        public NdArrayOpInfoListArguments AnonymousMember5;

        /// NDArrayOpInfo_declare_backward_dependency
        public NdArrayOpInfoDeclareBackwardDependency AnonymousMember6;

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
    public delegate bool CustomOpInfoForward(int param0, ref IntPtr param1, ref int param2, ref int param3, [MarshalAs(UnmanagedType.I1)] bool param4, IntPtr param5);

    /// Return Type: boolean
    ///param0: int
    ///param1: void**
    ///param2: int*
    ///param3: int*
    ///param4: boolean
    ///param5: void*
    public delegate bool CustomOpInfoBackward(int param0, ref IntPtr param1, ref int param2, ref int param3, [MarshalAs(UnmanagedType.I1)] bool param4, IntPtr param5);

    /// Return Type: boolean
    ///param0: void*
    public delegate bool CustomOpInfoDel(IntPtr param0);

    [StructLayout(LayoutKind.Sequential)]
    public struct CustomOpInfo
    {

        /// CustomOpInfo_forward
        public CustomOpInfoForward AnonymousMember1;

        /// CustomOpInfo_backward
        public CustomOpInfoBackward AnonymousMember2;

        /// CustomOpInfo_del
        public CustomOpInfoDel AnonymousMember3;

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
    public delegate bool CustomOpPropInfoListArguments(ref IntPtr param0, IntPtr param1);

    /// Return Type: boolean
    ///param0: char***
    ///param1: void*
    public delegate bool CustomOpPropInfoListOutputs(ref IntPtr param0, IntPtr param1);

    /// Return Type: boolean
    ///param0: int
    ///param1: int*
    ///param2: unsigned int**
    ///param3: void*
    public delegate bool CustomOpPropInfoInferShape(int param0, ref int param1, ref IntPtr param2, IntPtr param3);

    /// Return Type: boolean
    ///param0: int*
    ///param1: int*
    ///param2: int*
    ///param3: int*
    ///param4: int**
    ///param5: void*
    public delegate bool CustomOpPropInfoDeclareBackwardDependency(ref int param0, ref int param1, ref int param2, ref int param3, ref IntPtr param4, IntPtr param5);

    /// Return Type: boolean
    ///param0: char*
    ///param1: int
    ///param2: unsigned int**
    ///param3: int*
    ///param4: int*
    ///param5: CustomOpInfo*
    ///param6: void*
    public delegate bool CustomOpPropInfoCreateOperator([In] [MarshalAs(UnmanagedType.LPStr)] string param0, int param1, ref IntPtr param2, ref int param3, ref int param4, ref CustomOpInfo param5, IntPtr param6);

    /// Return Type: boolean
    ///param0: char***
    ///param1: void*
    public delegate bool CustomOpPropInfoListAuxiliaryStates(ref IntPtr param0, IntPtr param1);

    /// Return Type: boolean
    ///param0: void*
    public delegate bool CustomOpPropInfoDel(IntPtr param0);

    [StructLayout(LayoutKind.Sequential)]
    public struct CustomOpPropInfo
    {

        /// CustomOpPropInfo_list_arguments
        public CustomOpPropInfoListArguments AnonymousMember1;

        /// CustomOpPropInfo_list_outputs
        public CustomOpPropInfoListOutputs AnonymousMember2;

        /// CustomOpPropInfo_infer_shape
        public CustomOpPropInfoInferShape AnonymousMember3;

        /// CustomOpPropInfo_declare_backward_dependency
        public CustomOpPropInfoDeclareBackwardDependency AnonymousMember4;

        /// CustomOpPropInfo_create_operator
        public CustomOpPropInfoCreateOperator AnonymousMember5;

        /// CustomOpPropInfo_list_auxiliary_states
        public CustomOpPropInfoListAuxiliaryStates AnonymousMember6;

        /// CustomOpPropInfo_del
        public CustomOpPropInfoDel AnonymousMember7;

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


  
    /// Return Type: void
    ///key: int
    ///recv: NDArrayHandle->void*
    ///local: NDArrayHandle->void*
    ///handle: void*
    public delegate void MxkvStoreUpdater(int key, IntPtr recv, IntPtr local, IntPtr handle);

    /// Return Type: void
    ///head: int
    ///body: char*
    ///controller_handle: void*
    public delegate void MxkvStoreServerController(int head, [In()] [MarshalAs(UnmanagedType.LPStr)] string body, IntPtr controllerHandle);


    public static class NativeMethods
    {

        static NativeMethods()
        {
            if (!File.Exists("libmxnet.dll"))
            {
                LoadLibrary("x64\\cublas64_75.dll");
                LoadLibrary("x64\\cudart64_75.dll");
                LoadLibrary("x64\\cudnn64_4.dll");
                LoadLibrary("x64\\curand64_75.dll");
                LoadLibrary("x64\\libgcc_s_seh-1.dll");
                LoadLibrary("x64\\libgfortran-3.dll");
                LoadLibrary("x64\\libquadmath-0.dll");
                LoadLibrary("x64\\libopenblas.dll");
                LoadLibrary("x64\\libmxnet.dll");

            }
        }

        [DllImport("Kernel32.dll")]
        private static extern IntPtr LoadLibrary(string path);





        /// Return Type: char*
        [DllImport("libmxnet.dll", EntryPoint = "MXGetLastError")]
        public static extern IntPtr MXGetLastErrorNative();

        public static string MxGetLastError()
        {
            return Marshal.PtrToStringAnsi(MXGetLastErrorNative());
        }


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
        public static extern int MXNDArrayCreateNone(out IntPtr @out);


        /// Return Type: int
        ///shape: mx_uint*
        ///ndim: mx_uint->unsigned int
        ///dev_type: int
        ///dev_id: int
        ///delay_alloc: int
        ///out: NDArrayHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArrayCreate")]
        public static extern int MXNDArrayCreate(
              [MarshalAs(UnmanagedType.LPArray,ArraySubType = UnmanagedType.U4)]uint[] shape, 
              uint ndim, DeviceType devType,
              int devId, 
              int delayAlloc,
              out IntPtr @out);


        /// Return Type: int
        ///shape: mx_uint*
        ///ndim: mx_uint->unsigned int
        ///dev_type: int
        ///dev_id: int
        ///delay_alloc: int
        ///dtype: int
        ///out: NDArrayHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArrayCreateEx")]
        public static extern int MXNDArrayCreateEx(
            [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.U4)]uint[] shape,
            uint ndim,
            DeviceType devType,
            int devId, 
            int delayAlloc, 
            int dtype,
            out IntPtr @out);


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
        public static extern int MXNDArraySaveRawBytes(IntPtr handle, ref uint outSize, ref IntPtr outBuf);


        /// Return Type: int
        ///fname: char*
        ///num_args: mx_uint->unsigned int
        ///args: NDArrayHandle*
        ///keys: char**
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArraySave")]
        public static extern int MXNDArraySave(
            [In] [MarshalAs(UnmanagedType.LPStr)] string fname,
            uint numArgs,
            [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.SysInt)] IntPtr[] args,
            [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] keys);


        /// Return Type: int
        ///fname: char*
        ///out_size: mx_uint*
        ///out_arr: NDArrayHandle**
        ///out_name_size: mx_uint*
        ///out_names: char***
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArrayLoad")]
        public static extern int MXNDArrayLoad(
            [In] [MarshalAs(UnmanagedType.LPStr)] string fname,
            out uint outSize,
            out IntPtr outArr,
            out uint outNameSize,
            out IntPtr outNames);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///data: void*
        ///size: size_t->unsigned int
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArraySyncCopyFromCPU")]
        public static extern int MXNDArraySyncCopyFromCPU(IntPtr handle, [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.R4)] float[] data, uint size);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///data: void*
        ///size: size_t->unsigned int
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArraySyncCopyToCPU")]
        public static extern int MXNDArraySyncCopyToCPU(IntPtr handle, IntPtr data,uint size);


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
        public static extern int MXNDArraySlice(IntPtr handle, uint sliceBegin, uint sliceEnd, out IntPtr @out);


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
        public static extern int MXNDArrayReshape(IntPtr handle,
            int ndim,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I4)] int[] dims,
            out IntPtr @out);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///out_dim: mx_uint*
        ///out_pdata: mx_uint**
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArrayGetShape")]
        public static extern int MXNDArrayGetShape(IntPtr handle, out uint outDim, out IntPtr outPdata);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///out_pdata: mx_float**
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArrayGetData")]
        public static extern int MXNDArrayGetData(IntPtr handle, ref IntPtr outPdata);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///out_dtype: int*
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArrayGetDType")]
        public static extern int MXNDArrayGetDType(IntPtr handle, out int outDtype);


        /// Return Type: int
        ///handle: NDArrayHandle->void*
        ///out_dev_type: int*
        ///out_dev_id: int*
        [DllImport("libmxnet.dll", EntryPoint = "MXNDArrayGetContext")]
        public static extern int MXNDArrayGetContext(IntPtr handle, ref int outDevType, ref int outDevId);


        /// Return Type: int
        ///out_size: mx_uint*
        ///out_array: FunctionHandle**
        [DllImport("libmxnet.dll", EntryPoint = "MXListFunctions")]
        public static extern int MXListFunctions(ref uint outSize, ref IntPtr outArray);


        /// Return Type: int
        ///name: char*
        ///out: FunctionHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXGetFunction")]
        public static extern int MXGetFunction([In] [MarshalAs(UnmanagedType.LPStr)] string name, out IntPtr @out);


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
        public static extern int MXFuncGetInfo(IntPtr fun, ref IntPtr name, ref IntPtr description, ref uint numArgs, ref IntPtr argNames, ref IntPtr argTypeInfos, ref IntPtr argDescriptions, ref IntPtr returnType);


        /// Return Type: int
        ///fun: FunctionHandle->void*
        ///num_use_vars: mx_uint*
        ///num_scalars: mx_uint*
        ///num_mutate_vars: mx_uint*
        ///type_mask: int*
        [DllImport("libmxnet.dll", EntryPoint = "MXFuncDescribe")]
        public static extern int MXFuncDescribe(IntPtr fun, ref uint numUseVars, ref uint numScalars, ref uint numMutateVars, ref int typeMask);


        /// Return Type: int
        ///fun: FunctionHandle->void*
        ///use_vars: NDArrayHandle*
        ///scalar_args: mx_float*
        ///mutate_vars: NDArrayHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXFuncInvoke")]
        public static extern int MXFuncInvoke(IntPtr fun,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.SysInt)] IntPtr[] useVars,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.R4)] float[] scalarArgs,
            ref IntPtr mutateVars);


        /// Return Type: int
        ///fun: FunctionHandle->void*
        ///use_vars: NDArrayHandle*
        ///scalar_args: mx_float*
        ///mutate_vars: NDArrayHandle*
        ///num_params: int
        ///param_keys: char**
        ///param_vals: char**
        [DllImport("libmxnet.dll", EntryPoint = "MXFuncInvokeEx")]
        public static extern int MXFuncInvokeEx(IntPtr fun,
            ref IntPtr useVars,
             [In]  [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.R4)]  float[] scalarArgs,
            ref IntPtr mutateVars,
            int numParams,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] paramKeys,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] paramVals);


        /// Return Type: int
        ///out_size: mx_uint*
        ///out_array: AtomicSymbolCreator**
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolListAtomicSymbolCreators")]
        public static extern int MXSymbolListAtomicSymbolCreators(out uint outSize,
            out IntPtr outArrayPtr);


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
          [Out]out uint numArgs,
          [Out]out IntPtr argNames,
          [Out]out IntPtr argTypeInfos,
          [Out]out IntPtr argDescriptions,
          [Out]out IntPtr keyVarNumArgs,
          [Out]out IntPtr returnType);


        /// Return Type: int
        ///creator: AtomicSymbolCreator->void*
        ///num_param: mx_uint->unsigned int
        ///keys: char**
        ///vals: char**
        ///out: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolCreateAtomicSymbol")]
        public static extern int MXSymbolCreateAtomicSymbol(IntPtr creator, 
            uint numParam,
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
        public static extern int MXSymbolCreateGroup(uint numSymbols,
           [In]  [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.SysUInt)] IntPtr[] symbols, out IntPtr @out);


        /// Return Type: int
        ///fname: char*
        ///out: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolCreateFromFile")]
        public static extern int MXSymbolCreateFromFile([In] [MarshalAs(UnmanagedType.LPStr)] string fname, out IntPtr @out);


        /// Return Type: int
        ///json: char*
        ///out: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolCreateFromJSON")]
        public static extern int MXSymbolCreateFromJSON([In] [MarshalAs(UnmanagedType.LPStr)] string json, out IntPtr @out);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///fname: char*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolSaveToFile")]
        public static extern int MXSymbolSaveToFile(IntPtr symbol, [In] [MarshalAs(UnmanagedType.LPStr)] string fname);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out_json: char**
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolSaveToJSON")]
        public static extern int MXSymbolSaveToJSON(IntPtr symbol, out IntPtr outJson);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolFree")]
        public static extern int MXSymbolFree(IntPtr symbol);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolCopy")]
        public static extern int MXSymbolCopy(IntPtr symbol, out IntPtr @out);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out_str: char**
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolPrint")]
        public static extern int MXSymbolPrint(IntPtr symbol, ref IntPtr outStr);


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
        public static extern int MXSymbolListAttr(IntPtr symbol, out uint outSize, out IntPtr @out);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out_size: mx_uint*
        ///out: char***
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolListAttrShallow")]
        public static extern int MXSymbolListAttrShallow(IntPtr symbol, out uint outSize, out IntPtr @out);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out_size: mx_uint*
        ///out_str_array: char***
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolListArguments")]
        public static extern int MXSymbolListArguments(IntPtr symbol, out uint outSize, out IntPtr outStrArray);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out_size: mx_uint*
        ///out_str_array: char***
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolListOutputs")]
        public static extern int MXSymbolListOutputs(IntPtr symbol, out uint outSize, out IntPtr outStrArray);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolGetInternals")]
        public static extern int MXSymbolGetInternals(IntPtr symbol, out IntPtr @out);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///index: mx_uint->unsigned int
        ///out: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolGetOutput")]
        public static extern int MXSymbolGetOutput(IntPtr symbol, uint index, out IntPtr @out);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out_size: mx_uint*
        ///out_str_array: char***
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolListAuxiliaryStates")]
        public static extern int MXSymbolListAuxiliaryStates(IntPtr symbol, out uint outSize, out IntPtr outStrArray);


        /// Return Type: int
        ///sym: SymbolHandle->void*
        ///name: char*
        ///num_args: mx_uint->unsigned int
        ///keys: char**
        ///args: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolCompose")]
        public static extern int MXSymbolCompose(IntPtr sym, 
            [In] [MarshalAs(UnmanagedType.LPStr)] string name,
            uint numArgs,
           [In]  [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] keys,
           [In]  [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.SysUInt)] IntPtr[] args);

        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolCompose")]
        public static extern int MXSymbolCompose(IntPtr sym,
          [In] [MarshalAs(UnmanagedType.LPStr)] string name,
          uint numArgs,
         [In]  IntPtr keys,
         [In]  [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.SysUInt)] IntPtr[] args);


        /// Return Type: int
        ///sym: SymbolHandle->void*
        ///num_wrt: mx_uint->unsigned int
        ///wrt: char**
        ///out: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolGrad")]
        public static extern int MXSymbolGrad(IntPtr sym, uint numWrt, ref IntPtr wrt, ref IntPtr @out);


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
        public static extern int MXSymbolInferShape(IntPtr sym, 
            uint numArgs,
                  [In]  [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] keys,
         [In]  [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.U4)] uint[]     argIndPtr,
                [In]  [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.U4)] uint[] argShapeData, 
            out uint inShapeSize,
            out IntPtr inShapeNdim,
            out IntPtr inShapeData,
            out uint outShapeSize,
            out IntPtr outShapeNdim,
            out IntPtr outShapeData,
            out uint auxShapeSize,
            out IntPtr auxShapeNdim,
            out IntPtr auxShapeData,
            out int complete);




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
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolInferShapePartial")]
        public static extern int MXSymbolInferShapePartial(IntPtr sym, uint numArgs, ref IntPtr keys, ref uint argIndPtr, ref uint argShapeData, ref uint inShapeSize, ref IntPtr inShapeNdim, ref IntPtr inShapeData, ref uint outShapeSize, ref IntPtr outShapeNdim, ref IntPtr outShapeData, ref uint auxShapeSize, ref IntPtr auxShapeNdim, ref IntPtr auxShapeData, ref int complete);


        /// Return Type: int
        ///sym: SymbolHandle->void*
        ///num_args: mx_uint->unsigned int
        ///keys: char**
        ///arg_type_data: int*
        ///in_type_size: mx_uint*
        ///in_type_data: int**
        ///out_type_size: mx_uint*
        ///out_type_data: int**
        ///aux_type_size: mx_uint*
        ///aux_type_data: int**
        ///complete: int*
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolInferType")]
        public static extern int MXSymbolInferType(IntPtr sym,
            uint numArgs,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] keys,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I4)] int[] argTypeData,
            out uint inTypeSize,
            out IntPtr inTypeData,
            out uint outTypeSize,
            out IntPtr outTypeData,
            out uint auxTypeSize,
            out IntPtr auxTypeData,
            out int complete);


        /// Return Type: int
        ///handle: ExecutorHandle->void*
        [DllImport("libmxnet.dll", EntryPoint = "MXExecutorFree")]
        public static extern int MXExecutorFree(IntPtr handle);


        /// Return Type: int
        ///handle: ExecutorHandle->void*
        ///out_str: char**
        [DllImport("libmxnet.dll", EntryPoint = "MXExecutorPrint")]
        public static extern int MXExecutorPrint(IntPtr handle, out IntPtr outStr);


        /// Return Type: int
        ///handle: ExecutorHandle->void*
        ///is_train: int
        [DllImport("libmxnet.dll", EntryPoint = "MXExecutorForward")]
        public static extern int MXExecutorForward(IntPtr handle, int isTrain);


        /// Return Type: int
        ///handle: ExecutorHandle->void*
        ///len: mx_uint->unsigned int
        ///head_grads: NDArrayHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXExecutorBackward")]
        public static extern int MXExecutorBackward(IntPtr handle, uint len, [In]  [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.SysUInt)] IntPtr[] headGrads);

        /// Return Type: int
        ///handle: ExecutorHandle->void*
        ///len: mx_uint->unsigned int
        ///head_grads: NDArrayHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXExecutorBackward")]
        public static extern int MXExecutorBackward(IntPtr handle, uint len, IntPtr headGrads);

        /// Return Type: int
        ///handle: ExecutorHandle->void*
        ///out_size: mx_uint*
        ///out: NDArrayHandle**
        [DllImport("libmxnet.dll", EntryPoint = "MXExecutorOutputs")]
        public static extern int MXExecutorOutputs(IntPtr handle, out uint outSize, out IntPtr @out);


        /// Return Type: int
        ///symbol_handle: SymbolHandle->void*
        ///dev_type: int
        ///dev_id: int
        ///len: mx_uint->unsigned int
        ///in_args: NDArrayHandle*
        ///arg_grad_store: NDArrayHandle*
        ///grad_req_type: mx_uint*
        ///aux_states_len: mx_uint->unsigned int
        ///aux_states: NDArrayHandle*
        ///out: ExecutorHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXExecutorBind")]
        public static extern int MXExecutorBind(IntPtr symbolHandle, int devType, int devId, uint len, ref IntPtr inArgs, ref IntPtr argGradStore, ref uint gradReqType, uint auxStatesLen, ref IntPtr auxStates, ref IntPtr @out);


        /// Return Type: int
        ///symbol_handle: SymbolHandle->void*
        ///dev_type: int
        ///dev_id: int
        ///num_map_keys: mx_uint->unsigned int
        ///map_keys: char**
        ///map_dev_types: int*
        ///map_dev_ids: int*
        ///len: mx_uint->unsigned int
        ///in_args: NDArrayHandle*
        ///arg_grad_store: NDArrayHandle*
        ///grad_req_type: mx_uint*
        ///aux_states_len: mx_uint->unsigned int
        ///aux_states: NDArrayHandle*
        ///out: ExecutorHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXExecutorBindX")]
        public static extern int MXExecutorBindX(IntPtr symbolHandle, int devType, int devId, uint numMapKeys, ref IntPtr mapKeys, ref int mapDevTypes, ref int mapDevIds, uint len, ref IntPtr inArgs, ref IntPtr argGradStore, ref uint gradReqType, uint auxStatesLen, ref IntPtr auxStates, ref IntPtr @out);


        /// Return Type: int
        ///symbol_handle: SymbolHandle->void*
        ///dev_type: int
        ///dev_id: int
        ///num_map_keys: mx_uint->unsigned int
        ///map_keys: char**
        ///map_dev_types: int*
        ///map_dev_ids: int*
        ///len: mx_uint->unsigned int
        ///in_args: NDArrayHandle*
        ///arg_grad_store: NDArrayHandle*
        ///grad_req_type: mx_uint*
        ///aux_states_len: mx_uint->unsigned int
        ///aux_states: NDArrayHandle*
        ///shared_exec: ExecutorHandle->void*
        ///out: ExecutorHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXExecutorBindEX")]
        public static extern int MXExecutorBindEX(IntPtr symbolHandle,
            int devType,
            int devId,
            uint numMapKeys,
            [In]  [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] mapKeys,
            [In]  [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I4)] int[] mapDevTypes,
            [In]  [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I4)] int[] mapDevIds,
            uint len,
            [In]  [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.SysUInt)] IntPtr[] inArgs,
            [In]  [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.SysUInt)] IntPtr[] argGradStore,
            [In]  [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.U4)] uint[] gradReqType,
            uint auxStatesLen,
            [In]  [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.SysUInt)] IntPtr[] auxStates,
            IntPtr sharedExec,
            out IntPtr @out);

        /// Return Type: int
        ///handle: ExecutorHandle->void*
        ///callback: ExecutorMonitorCallback
        ///callback_handle: void*
        [DllImport("libmxnet.dll", EntryPoint = "MXExecutorSetMonitorCallback")]
        public static extern int MXExecutorSetMonitorCallback(IntPtr handle, ExecutorMonitorCallback callback, IntPtr callbackHandle);


        /// Return Type: int
        ///out_size: mx_uint*
        ///out_array: DataIterCreator**
        [DllImport("libmxnet.dll", EntryPoint = "MXListDataIters")]
        public static extern int MXListDataIters(ref uint outSize, ref IntPtr outArray);


        /// Return Type: int
        ///handle: DataIterCreator->void*
        ///num_param: mx_uint->unsigned int
        ///keys: char**
        ///vals: char**
        ///out: DataIterHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXDataIterCreateIter")]
        public static extern int MXDataIterCreateIter(IntPtr handle, uint numParam, ref IntPtr keys, ref IntPtr vals, ref IntPtr @out);


        /// Return Type: int
        ///creator: DataIterCreator->void*
        ///name: char**
        ///description: char**
        ///num_args: mx_uint*
        ///arg_names: char***
        ///arg_type_infos: char***
        ///arg_descriptions: char***
        [DllImport("libmxnet.dll", EntryPoint = "MXDataIterGetIterInfo")]
        public static extern int MXDataIterGetIterInfo(IntPtr creator, ref IntPtr name, ref IntPtr description, ref uint numArgs, ref IntPtr argNames, ref IntPtr argTypeInfos, ref IntPtr argDescriptions);


        /// Return Type: int
        ///handle: DataIterHandle->void*
        [DllImport("libmxnet.dll", EntryPoint = "MXDataIterFree")]
        public static extern int MXDataIterFree(IntPtr handle);


        /// Return Type: int
        ///handle: DataIterHandle->void*
        ///out: int*
        [DllImport("libmxnet.dll", EntryPoint = "MXDataIterNext")]
        public static extern int MXDataIterNext(IntPtr handle, ref int @out);


        /// Return Type: int
        ///handle: DataIterHandle->void*
        [DllImport("libmxnet.dll", EntryPoint = "MXDataIterBeforeFirst")]
        public static extern int MXDataIterBeforeFirst(IntPtr handle);


        /// Return Type: int
        ///handle: DataIterHandle->void*
        ///out: NDArrayHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXDataIterGetData")]
        public static extern int MXDataIterGetData(IntPtr handle, ref IntPtr @out);


        /// Return Type: int
        ///handle: DataIterHandle->void*
        ///out_index: uint64_t**
        ///out_size: uint64_t*
        [DllImport("libmxnet.dll", EntryPoint = "MXDataIterGetIndex")]
        public static extern int MXDataIterGetIndex(IntPtr handle, ref IntPtr outIndex, ref ulong outSize);


        /// Return Type: int
        ///handle: DataIterHandle->void*
        ///pad: int*
        [DllImport("libmxnet.dll", EntryPoint = "MXDataIterGetPadNum")]
        public static extern int MXDataIterGetPadNum(IntPtr handle, ref int pad);


        /// Return Type: int
        ///handle: DataIterHandle->void*
        ///out: NDArrayHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXDataIterGetLabel")]
        public static extern int MXDataIterGetLabel(IntPtr handle, ref IntPtr @out);


        /// Return Type: int
        ///num_vars: mx_uint->unsigned int
        ///keys: char**
        ///vals: char**
        [DllImport("libmxnet.dll", EntryPoint = "MXInitPSEnv")]
        public static extern int MXInitPSEnv(uint numVars, ref IntPtr keys, ref IntPtr vals);


        /// Return Type: int
        ///type: char*
        ///out: KVStoreHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXKVStoreCreate")]
        public static extern int MXKVStoreCreate([In()] [MarshalAs(UnmanagedType.LPStr)] string type, out IntPtr @out);


        /// Return Type: int
        ///handle: KVStoreHandle->void*
        [DllImport("libmxnet.dll", EntryPoint = "MXKVStoreFree")]
        public static extern int MXKVStoreFree(IntPtr handle);


        /// Return Type: int
        ///handle: KVStoreHandle->void*
        ///num: mx_uint->unsigned int
        ///keys: int*
        ///vals: NDArrayHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXKVStoreInit")]
        public static extern int MXKVStoreInit(IntPtr handle, uint num,
                 [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I4)] int[]   keys,
                 [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.SysInt)] IntPtr[] vals);


        /// Return Type: int
        /// handle: KVStoreHandle->void*
        /// num: mx_uint->unsigned int
        /// keys: int*
        /// vals: NDArrayHandle*
        /// priority: int
        [DllImport("libmxnet.dll", EntryPoint = "MXKVStorePush")]
        public static extern int MXKVStorePush(IntPtr handle, uint num,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I4)] int[] keys,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.SysInt)] IntPtr[] vals
            , int priority);


        /// Return Type: int
        ///handle: KVStoreHandle->void*
        ///num: mx_uint->unsigned int
        ///keys: int*
        ///vals: NDArrayHandle*
        ///priority: int
        [DllImport("libmxnet.dll", EntryPoint = "MXKVStorePull")]
        public static extern int MXKVStorePull(IntPtr handle, uint num,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I4)] int[] keys,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.SysInt)] IntPtr[] vals, int priority);


        /// Return Type: int
        ///handle: KVStoreHandle->void*
        ///updater: MXKVStoreUpdater
        ///updater_handle: void*
        [DllImport("libmxnet.dll", EntryPoint = "MXKVStoreSetUpdater")]
        public static extern int MXKVStoreSetUpdater(IntPtr handle,
            [MarshalAs(UnmanagedType.FunctionPtr)]MxkvStoreUpdater updater,
            IntPtr updaterHandle);


        /// Return Type: int
        ///handle: KVStoreHandle->void*
        ///type: char**
        [DllImport("libmxnet.dll", EntryPoint = "MXKVStoreGetType")]
        public static extern int MXKVStoreGetType(IntPtr handle, out IntPtr type);


        /// Return Type: int
        ///handle: KVStoreHandle->void*
        ///ret: int*
        [DllImport("libmxnet.dll", EntryPoint = "MXKVStoreGetRank")]
        public static extern int MXKVStoreGetRank(IntPtr handle, out int ret);


        /// Return Type: int
        ///handle: KVStoreHandle->void*
        ///ret: int*
        [DllImport("libmxnet.dll", EntryPoint = "MXKVStoreGetGroupSize")]
        public static extern int MXKVStoreGetGroupSize(IntPtr handle, out int ret);


        /// Return Type: int
        ///ret: int*
        [DllImport("libmxnet.dll", EntryPoint = "MXKVStoreIsWorkerNode")]
        public static extern int MXKVStoreIsWorkerNode(out int ret);


        /// Return Type: int
        ///ret: int*
        [DllImport("libmxnet.dll", EntryPoint = "MXKVStoreIsServerNode")]
        public static extern int MXKVStoreIsServerNode(out int ret);


        /// Return Type: int
        ///ret: int*
        [DllImport("libmxnet.dll", EntryPoint = "MXKVStoreIsSchedulerNode")]
        public static extern int MXKVStoreIsSchedulerNode(out int ret);


        /// Return Type: int
        ///handle: KVStoreHandle->void*
        [DllImport("libmxnet.dll", EntryPoint = "MXKVStoreBarrier")]
        public static extern int MXKVStoreBarrier(IntPtr handle);


        /// Return Type: int
        ///handle: KVStoreHandle->void*
        ///barrier_before_exit: int
        [DllImport("libmxnet.dll", EntryPoint = "MXKVStoreSetBarrierBeforeExit")]
        public static extern int MXKVStoreSetBarrierBeforeExit(IntPtr handle, int barrierBeforeExit);


        /// Return Type: int
        ///handle: KVStoreHandle->void*
        ///controller: MXKVStoreServerController
        ///controller_handle: void*
        [DllImport("libmxnet.dll", EntryPoint = "MXKVStoreRunServer")]
        public static extern int MXKVStoreRunServer(IntPtr handle, MxkvStoreServerController controller, IntPtr controllerHandle);


        /// Return Type: int
        ///handle: KVStoreHandle->void*
        ///cmd_id: int
        ///cmd_body: char*
        [DllImport("libmxnet.dll", EntryPoint = "MXKVStoreSendCommmandToServers")]
        public static extern int MXKVStoreSendCommmandToServers(IntPtr handle, int cmdId,
            [In(), MarshalAs(UnmanagedType.LPStr)]  string cmdBody);


        /// Return Type: int
        ///handle: KVStoreHandle->void*
        ///node_id: int
        ///number: int*
        ///timeout_sec: int
        [DllImport("libmxnet.dll", EntryPoint = "MXKVStoreGetNumDeadNode")]
        public static extern int MXKVStoreGetNumDeadNode(IntPtr handle, int nodeId, ref int number, int timeoutSec);


        /// Return Type: int
        ///uri: char*
        ///out: RecordIOHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXRecordIOWriterCreate")]
        public static extern int MXRecordIOWriterCreate([In()] [MarshalAs(UnmanagedType.LPStr)] string uri, ref IntPtr @out);


        /// Return Type: int
        ///handle: RecordIOHandle->void*
        [DllImport("libmxnet.dll", EntryPoint = "MXRecordIOWriterFree")]
        public static extern int MXRecordIOWriterFree(IntPtr handle);


        /// Return Type: int
        ///handle: RecordIOHandle*
        ///buf: char*
        ///size: size_t->unsigned int
        [DllImport("libmxnet.dll", EntryPoint = "MXRecordIOWriterWriteRecord")]
        public static extern int MXRecordIOWriterWriteRecord(ref IntPtr handle, [In()] [MarshalAs(UnmanagedType.LPStr)] string buf, [MarshalAs(UnmanagedType.SysUInt)] uint size);


        /// Return Type: int
        ///handle: RecordIOHandle*
        ///pos: size_t*
        [DllImport("libmxnet.dll", EntryPoint = "MXRecordIOWriterTell")]
        public static extern int MXRecordIOWriterTell(ref IntPtr handle, ref uint pos);


        /// Return Type: int
        ///uri: char*
        ///out: RecordIOHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXRecordIOReaderCreate")]
        public static extern int MXRecordIOReaderCreate([In()] [MarshalAs(UnmanagedType.LPStr)] string uri, ref IntPtr @out);


        /// Return Type: int
        ///handle: RecordIOHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXRecordIOReaderFree")]
        public static extern int MXRecordIOReaderFree(ref IntPtr handle);


        /// Return Type: int
        ///handle: RecordIOHandle*
        ///buf: char**
        ///size: size_t*
        [DllImport("libmxnet.dll", EntryPoint = "MXRecordIOReaderReadRecord")]
        public static extern int MXRecordIOReaderReadRecord(ref IntPtr handle, ref IntPtr buf, ref uint size);


        /// Return Type: int
        ///handle: RecordIOHandle*
        ///pos: size_t->unsigned int
        [DllImport("libmxnet.dll", EntryPoint = "MXRecordIOReaderSeek")]
        public static extern int MXRecordIOReaderSeek(ref IntPtr handle, [MarshalAs(UnmanagedType.SysUInt)] uint pos);


        /// Return Type: int
        ///name: char*
        ///num_input: mx_uint->unsigned int
        ///num_output: mx_uint->unsigned int
        ///input_names: char**
        ///output_names: char**
        ///inputs: NDArrayHandle*
        ///outputs: NDArrayHandle*
        ///kernel: char*
        ///out: RtcHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXRtcCreate")]
        public static extern int MXRtcCreate(IntPtr name, uint numInput, uint numOutput, ref IntPtr inputNames, ref IntPtr outputNames, ref IntPtr inputs, ref IntPtr outputs, IntPtr kernel, ref IntPtr @out);


        /// Return Type: int
        ///handle: RtcHandle->void*
        ///num_input: mx_uint->unsigned int
        ///num_output: mx_uint->unsigned int
        ///inputs: NDArrayHandle*
        ///outputs: NDArrayHandle*
        ///gridDimX: mx_uint->unsigned int
        ///gridDimY: mx_uint->unsigned int
        ///gridDimZ: mx_uint->unsigned int
        ///blockDimX: mx_uint->unsigned int
        ///blockDimY: mx_uint->unsigned int
        ///blockDimZ: mx_uint->unsigned int
        [DllImport("libmxnet.dll", EntryPoint = "MXRtcPush")]
        public static extern int MXRtcPush(IntPtr handle, uint numInput, uint numOutput, ref IntPtr inputs, ref IntPtr outputs, uint gridDimX, uint gridDimY, uint gridDimZ, uint blockDimX, uint blockDimY, uint blockDimZ);


        /// Return Type: int
        ///handle: RtcHandle->void*
        [DllImport("libmxnet.dll", EntryPoint = "MXRtcFree")]
        public static extern int MXRtcFree(IntPtr handle);


        /// Return Type: int
        ///key: char*
        ///out: OptimizerCreator*
        [DllImport("libmxnet.dll", EntryPoint = "MXOptimizerFindCreator")]
        public static extern int MXOptimizerFindCreator([In()] [MarshalAs(UnmanagedType.LPStr)] string key, out IntPtr @out);


        /// Return Type: int
        ///creator: OptimizerCreator->void*
        ///num_param: mx_uint->unsigned int
        ///keys: char**
        ///vals: char**
        ///out: OptimizerHandle*
        [DllImport("libmxnet.dll", EntryPoint = "MXOptimizerCreateOptimizer")]
        public static extern int MXOptimizerCreateOptimizer(IntPtr creator, uint numParam,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] keys,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] vals,
            out IntPtr @out);


        /// Return Type: int
        ///handle: OptimizerHandle->void*
        [DllImport("libmxnet.dll", EntryPoint = "MXOptimizerFree")]
        public static extern int MXOptimizerFree(IntPtr handle);


        /// Return Type: int
        ///handle: OptimizerHandle->void*
        ///index: int
        ///weight: NDArrayHandle->void*
        ///grad: NDArrayHandle->void*
        ///lr: mx_float->float
        ///wd: mx_float->float
        [DllImport("libmxnet.dll", EntryPoint = "MXOptimizerUpdate")]
        public static extern int MXOptimizerUpdate(IntPtr handle, int index, IntPtr weight, IntPtr grad, float lr, float wd);


        /// Return Type: int
        ///op_type: char*
        ///creator: CustomOpPropCreator
        [DllImport("libmxnet.dll", EntryPoint = "MXCustomOpRegister")]
        public static extern int MXCustomOpRegister([In()] [MarshalAs(UnmanagedType.LPStr)] string opType, CustomOpPropCreator creator);


        /// Return Type: int
        /// creator: AtomicSymbolCreator->void*
        /// num_inputs: int
        /// inputs: NDArrayHandle*
        /// num_outputs: int*
        /// outputs: NDArrayHandle**
        /// num_params: int
        /// param_keys: char**
        /// param_vals: char**
        [DllImport("libmxnet.dll", EntryPoint = "MXImperativeInvoke")]
        public static extern int MXImperativeInvoke(IntPtr creator,
            int num_inputs,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.SysInt)] IntPtr[] inputs,
            ref int num_outputs,
            ref IntPtr outputs,
            int num_params,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] param_keys,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] param_vals);



        #region NNVM
        /// Return Type: void
        ///msg: char*
        [DllImport("libmxnet.dll", EntryPoint = "NNAPISetLastError")]
        public static extern void NNAPISetLastError([In()] [MarshalAs(UnmanagedType.LPStr)] string msg);


        /// Return Type: char*
        [DllImport("libmxnet.dll", EntryPoint = "NNGetLastError")]
        public static extern IntPtr NNGetLastErrorNative();

        public static string NnGetLastError()
        {
            return Marshal.PtrToStringAnsi(NNGetLastErrorNative());
        }


        /// Return Type: int
        ///out_size: nn_uint*
        ///out_array: char***
        [DllImport("libmxnet.dll", EntryPoint = "NNListAllOpNames")]
        public static extern int NNListAllOpNames(out uint out_size, out IntPtr out_array);


        /// Return Type: int
        ///op_name: char*
        ///op_out: OpHandle*
        [DllImport("libmxnet.dll", EntryPoint = "NNGetOpHandle")]
        public static extern int NNGetOpHandle([In()] [MarshalAs(UnmanagedType.LPStr)] string op_name, out IntPtr op_out);


        /// Return Type: int
        ///out_size: nn_uint*
        ///out_array: OpHandle**
        [DllImport("libmxnet.dll", EntryPoint = "NNListUniqueOps")]
        public static extern int NNListUniqueOps(ref uint out_size, ref IntPtr out_array);


        /// Return Type: int
        ///op: OpHandle->void*
        ///real_name: char**
        ///description: char**
        ///num_doc_args: nn_uint*
        ///arg_names: char***
        ///arg_type_infos: char***
        ///arg_descriptions: char***
        ///return_type: char**
        [DllImport("libmxnet.dll", EntryPoint = "NNGetOpInfo")]
        public static extern int NNGetOpInfo(IntPtr op, ref IntPtr real_name, ref IntPtr description, ref uint num_doc_args, ref IntPtr arg_names, ref IntPtr arg_type_infos, ref IntPtr arg_descriptions, ref IntPtr return_type);


        /// Return Type: int
        ///op: OpHandle->void*
        ///num_param: nn_uint->unsigned int
        ///keys: char**
        ///vals: char**
        ///out: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "NNSymbolCreateAtomicSymbol")]
        public static extern int NNSymbolCreateAtomicSymbol(IntPtr op, uint num_param, ref IntPtr keys, ref IntPtr vals, ref IntPtr @out);


        /// Return Type: int
        ///name: char*
        ///out: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "NNSymbolCreateVariable")]
        public static extern int NNSymbolCreateVariable([In()] [MarshalAs(UnmanagedType.LPStr)] string name, ref IntPtr @out);


        /// Return Type: int
        ///num_symbols: nn_uint->unsigned int
        ///symbols: SymbolHandle*
        ///out: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "NNSymbolCreateGroup")]
        public static extern int NNSymbolCreateGroup(uint num_symbols, ref IntPtr symbols, ref IntPtr @out);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        [DllImport("libmxnet.dll", EntryPoint = "NNSymbolFree")]
        public static extern int NNSymbolFree(IntPtr symbol);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "NNSymbolCopy")]
        public static extern int NNSymbolCopy(IntPtr symbol, ref IntPtr @out);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out_str: char**
        [DllImport("libmxnet.dll", EntryPoint = "NNSymbolPrint")]
        public static extern int NNSymbolPrint(IntPtr symbol, ref IntPtr out_str);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///key: char*
        ///out: char**
        ///success: int*
        [DllImport("libmxnet.dll", EntryPoint = "NNSymbolGetAttr")]
        public static extern int NNSymbolGetAttr(IntPtr symbol, [In()] [MarshalAs(UnmanagedType.LPStr)] string key, ref IntPtr @out, ref int success);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///num_param: nn_uint->unsigned int
        ///keys: char**
        ///values: char**
        [DllImport("libmxnet.dll", EntryPoint = "NNSymbolSetAttrs")]
        public static extern int NNSymbolSetAttrs(IntPtr symbol, uint num_param, ref IntPtr keys, ref IntPtr values);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///recursive_option: int
        ///out_size: nn_uint*
        ///out: char***
        [DllImport("libmxnet.dll", EntryPoint = "NNSymbolListAttrs")]
        public static extern int NNSymbolListAttrs(IntPtr symbol, int recursive_option, ref uint out_size, ref IntPtr @out);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///option: int
        ///out_size: nn_uint*
        ///out_str_array: char***
        [DllImport("libmxnet.dll", EntryPoint = "NNSymbolListInputNames")]
        public static extern int NNSymbolListInputNames(IntPtr symbol, int option, ref uint out_size, ref IntPtr out_str_array);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out_size: nn_uint*
        ///out_str_array: char***
        [DllImport("libmxnet.dll", EntryPoint = "NNSymbolListOutputNames")]
        public static extern int NNSymbolListOutputNames(IntPtr symbol, ref uint out_size, ref IntPtr out_str_array);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///out: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "NNSymbolGetInternals")]
        public static extern int NNSymbolGetInternals(IntPtr symbol, ref IntPtr @out);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///index: nn_uint->unsigned int
        ///out: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "NNSymbolGetOutput")]
        public static extern int NNSymbolGetOutput(IntPtr symbol, uint index, ref IntPtr @out);


        /// Return Type: int
        ///sym: SymbolHandle->void*
        ///name: char*
        ///num_args: nn_uint->unsigned int
        ///keys: char**
        ///args: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "NNSymbolCompose")]
        public static extern int NNSymbolCompose(IntPtr sym, [In()] [MarshalAs(UnmanagedType.LPStr)] string name, uint num_args, ref IntPtr keys, ref IntPtr args);


        /// Return Type: int
        ///symbol: SymbolHandle->void*
        ///graph: GraphHandle*
        [DllImport("libmxnet.dll", EntryPoint = "NNGraphCreate")]
        public static extern int NNGraphCreate(IntPtr symbol, ref IntPtr graph);


        /// Return Type: int
        ///handle: GraphHandle->void*
        [DllImport("libmxnet.dll", EntryPoint = "NNGraphFree")]
        public static extern int NNGraphFree(IntPtr handle);


        /// Return Type: int
        ///graph: GraphHandle->void*
        ///symbol: SymbolHandle*
        [DllImport("libmxnet.dll", EntryPoint = "NNGraphGetSymbol")]
        public static extern int NNGraphGetSymbol(IntPtr graph, ref IntPtr symbol);


        /// Return Type: int
        ///handle: GraphHandle->void*
        ///key: char*
        ///json_value: char*
        [DllImport("libmxnet.dll", EntryPoint = "NNGraphSetJSONAttr")]
        public static extern int NNGraphSetJSONAttr(IntPtr handle, [In()] [MarshalAs(UnmanagedType.LPStr)] string key, [In()] [MarshalAs(UnmanagedType.LPStr)] string json_value);


        /// Return Type: int
        ///handle: SymbolHandle->void*
        ///key: char*
        ///json_out: char**
        ///success: int*
        [DllImport("libmxnet.dll", EntryPoint = "NNGraphGetJSONAttr")]
        public static extern int NNGraphGetJSONAttr(IntPtr handle, [In()] [MarshalAs(UnmanagedType.LPStr)] string key, ref IntPtr json_out, ref int success);


        /// Return Type: int
        ///handle: GraphHandle->void*
        ///key: char*
        ///list: SymbolHandle->void*
        [DllImport("libmxnet.dll", EntryPoint = "NNGraphSetNodeEntryListAttr_")]
        public static extern int NNGraphSetNodeEntryListAttr_(IntPtr handle, [In()] [MarshalAs(UnmanagedType.LPStr)] string key, IntPtr list);


        /// Return Type: int
        ///src: GraphHandle->void*
        ///num_pass: nn_uint->unsigned int
        ///pass_names: char**
        ///dst: GraphHandle*
        [DllImport("libmxnet.dll", EntryPoint = "NNGraphApplyPasses")]
        public static extern int NNGraphApplyPasses(IntPtr src, uint num_pass, ref IntPtr pass_names, ref IntPtr dst);


        #endregion


    }

}
