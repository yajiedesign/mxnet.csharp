using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using SymbolHandle = System.IntPtr;

namespace mxnet.csharp
{
    public class SymBlob : IDisposable
    {
        /// <summary>
        ///     default constructor
        /// </summary>
        private SymBlob()
        {
        }

        /// <summary>
        ///     construct with SymbolHandle to store
        /// </summary>
        /// <param name="handle"></param>
        public SymBlob(SymbolHandle handle)

        {
            Handle = handle;
        }

        /// <summary>
        ///     destructor, free the SymbolHandle
        /// </summary>
        ~SymBlob()
        {
            Dispose(false);
        }

        private void Dispose(bool disposing)
        {
            NativeMethods.MXSymbolFree(Handle);
            if (disposing)
            {
                GC.SuppressFinalize(this);
            }
        }

        public void Dispose()
        {
            Dispose(true);
        }

        /// <summary>
        ///     the SymbolHandle to store
        /// </summary>
        public SymbolHandle Handle { get; } = IntPtr.Zero;
    }

    public class Symbol : OperatorWarp
    {
        private readonly SymBlob _blobPtr;

        public Symbol(string name)
        {
            SymbolHandle symbolHandle;
            NativeMethods.MXSymbolCreateVariable(name, out symbolHandle);
            _blobPtr = new SymBlob(symbolHandle);
        }

        public Symbol(SymbolHandle symbolHandle)
        {
            _blobPtr = new SymBlob(symbolHandle);
        }

        public static Symbol Variable(string name)
        {
            return new Symbol(name);
        }


        public static Symbol operator +(Symbol lhs, Symbol rhs)
        {
            return Operator._Plus(lhs, rhs);
        }

        public static Symbol operator -(Symbol lhs, Symbol rhs)
        {
            return Operator._Minus(lhs, rhs);
        }

        public static Symbol operator *(Symbol lhs, Symbol rhs)
        {
            return Operator._Mul(lhs, rhs);
        }

        public static Symbol operator /(Symbol lhs, Symbol rhs)
        {
            return Operator._Div(lhs, rhs);
        }

        public static Symbol operator +(Symbol lhs, float scalar)
        {
            return Operator._PlusScalar(lhs, scalar);
        }

        public static Symbol operator +(float scalar, Symbol rhs)
        {
            return Operator._PlusScalar(rhs, scalar);
        }

        public static Symbol operator -(Symbol lhs, float scalar)
        {
            return Operator._MinusScalar(lhs, scalar);
        }

        public static Symbol operator -(float scalar, Symbol rhs)
        {
            return Operator._RMinusScalar(scalar, rhs);
        }

        public static Symbol operator *(Symbol lhs, float scalar)
        {
            return Operator._MulScalar(lhs, scalar);
        }

        public static Symbol operator *(float scalar, Symbol rhs)
        {
            return Operator._MulScalar(rhs, scalar);
        }

        public static Symbol operator /(Symbol lhs, float scalar)
        {
            return Operator._DivScalar(lhs, scalar);
        }

        public static Symbol operator /(float scalar, Symbol rhs)
        {
            return Operator._RDivScalar(scalar, rhs);
        }


        public Symbol this[int index]
        {
            get
            {
                SymbolHandle @out;
                NativeMethods.MXSymbolGetOutput(GetHandle(), (uint)index, out @out);
                return new Symbol(@out);
            }
        }

        public Symbol this[string index]
        {
            get
            {
                var outputs = ListOutputs();
                for (var i = 0; i < outputs.Count; i++)
                {
                    if (outputs[i] == index)
                    {
                        return this[i];
                    }
                }
                return this[0];
            }
        }

        public Symbol Group(IEnumerable<Symbol> symbols)
        {
            var handleList = new List<SymbolHandle>();
            foreach (var symbol in symbols)
            {
                handleList.Add(symbol.GetHandle());
            }
            SymbolHandle @out;

            NativeMethods.MXSymbolCreateGroup((uint)handleList.Count, handleList.ToArray(), out @out);
            return new Symbol(@out);
        }

        public Symbol Load(string fileName)
        {
            SymbolHandle handle;
            Debug.Assert(NativeMethods.MXSymbolCreateFromFile(fileName, out handle) == 0);
            return new Symbol(handle);
        }

        public Symbol LoadJson(string jsonStr)
        {
            SymbolHandle handle;
            Debug.Assert(NativeMethods.MXSymbolCreateFromJSON(jsonStr, out handle) == 0);
            return new Symbol(handle);
        }

        private void Save(string fileName)
        {
            Debug.Assert(NativeMethods.MXSymbolSaveToFile(GetHandle(), fileName) == 0);
        }

        public string ToJson()
        {
            IntPtr outJson;
            Debug.Assert(NativeMethods.MXSymbolSaveToJSON(GetHandle(), out outJson) == 0);
            return Marshal.PtrToStringAnsi(outJson);
        }

        public Symbol GetInternals()
        {
            SymbolHandle handle;
            Debug.Assert(NativeMethods.MXSymbolGetInternals(GetHandle(), out handle) == 0);
            return new Symbol(handle);
        }

        public Symbol Copy()
        {
            SymbolHandle handle;
            Debug.Assert(NativeMethods.MXSymbolCopy(GetHandle(), out handle) == 0);
            return new Symbol(handle);
        }

        public IList<string> ListArguments()
        {
            var ret = new List<string>();
            uint size;
            IntPtr sarrPtr;

            NativeMethods.MXSymbolListArguments(GetHandle(), out size, out sarrPtr);
            var sarr = new IntPtr[size];
            if (size > 0)
            {
                Marshal.Copy(sarrPtr, sarr, 0, (int)size);
            }
            for (var i = 0; i < size; i++)
            {
                ret.Add(Marshal.PtrToStringAnsi(sarr[i]));
            }
            return ret;
        }

        public IList<string> ListOutputs()
        {
            var ret = new List<string>();
            uint size;
            IntPtr sarrPtr;

            NativeMethods.MXSymbolListOutputs(GetHandle(), out size, out sarrPtr);
            var sarr = new IntPtr[size];
            if (size > 0)
            {
                Marshal.Copy(sarrPtr, sarr, 0, (int)size);
            }
            for (var i = 0; i < size; i++)
            {
                ret.Add(Marshal.PtrToStringAnsi(sarr[i]));
            }
            return ret;
        }

        public IList<string> ListAuxiliaryStates()
        {
            var ret = new List<string>();
            uint size;
            IntPtr sarrPtr;

            NativeMethods.MXSymbolListAuxiliaryStates(GetHandle(), out size, out sarrPtr);
            var sarr = new IntPtr[size];
            if (size > 0)
            {
                Marshal.Copy(sarrPtr, sarr, 0, (int)size);
            }
            for (var i = 0; i < size; i++)
            {
                ret.Add(Marshal.PtrToStringAnsi(sarr[i]));
            }
            return ret;
        }
        private static int[] PtrToArrayInt32(IntPtr ptr, int size)
        {
            if (size == 0)
            {
                return null;
            }
            int[] array = new int[size];
            Marshal.Copy(ptr, array, 0, size);
            return array;
        }

        private static uint[] PtrToArrayUint32(IntPtr ptr, int size)
        {
            if (size == 0)
            {
                return null;
            }
            int[] array = new int[size];
            Marshal.Copy(ptr, array, 0, size);
            var ret = array.Select(s => (uint)s).ToArray();

            return ret;
        }
        private static List<uint[]> PtrToArrayUint32(IntPtr dataPtr, int dataSize, uint[] shapeNdim)
        {
            if (dataSize == 0)
            {
                return null;
            }
            IntPtr[] ptrArray = new IntPtr[dataSize];
            Marshal.Copy(dataPtr, ptrArray, 0, dataSize);
            List<uint[]> ret = new List<uint[]>();

            for (int i = 0; i < dataSize; i++)
            {
                int[] data = new int[(int)shapeNdim[i]];
                Marshal.Copy(ptrArray[i], data, 0, (int)shapeNdim[i]);
                ret.Add(data.Select(s => (uint)s).ToArray());
            }

            return ret;
        }
        /// <summary>
        /// Infer the shape of outputs and arguments of given known shapes of arguments
        /// </summary>
        /// <param name="argShapes"> Provide keyword arguments of known shapes.</param>
        /// <param name="inShape">List of shapes of arguments.The order is in the same order as list_arguments()</param>
        /// <param name="auxShape">List of shapes of outputs.The order is in the same order as list_outputs()</param>
        /// <param name="outShape">List of shapes of outputs.The order is in the same order as list_auxiliary()</param>
        public void InferShape(Dictionary<string, Shape> argShapes, List<uint[]> inShape, List<uint[]> auxShape, List<uint[]> outShape)
        {
            InferShape(argShapes.ToDictionary(x=>x.Key,y=> (uint[])y.Value) , inShape, auxShape, outShape);
        }


        public void InferType(
         Dictionary<string, Type> input_types,
         [Out] List<Type> inType,
         [Out] List<Type> auxType,
         [Out] List<Type> outType)
        {
            var keys = new List<string>();
            var arg_type_data = new List<int>();
  

            foreach (var arg in input_types)
            {
                keys.Add(arg.Key);
                arg_type_data.Add( Util._DTYPE_NP_TO_MX[arg.Value]);
            }


            uint inTypeSize;
            IntPtr inTypeDataPtr;
    
            uint outTypeSize;
            IntPtr outTypeDataPtr;

            uint auxTypeSize;
            IntPtr auxTypeDataPtr;

            int complete;

            Debug.Assert(NativeMethods.MXSymbolInferType(GetHandle(), (uint)keys.Count, keys.ToArray(),
                arg_type_data.ToArray(),
                out inTypeSize, out inTypeDataPtr,
                out outTypeSize, out outTypeDataPtr,
                out auxTypeSize, out auxTypeDataPtr,
                out complete) ==
                         0, NativeMethods.MXGetLastError());

            var inTypeData = PtrToArrayInt32(inTypeDataPtr, (int)inTypeSize);
            var outTypeData = PtrToArrayInt32(outTypeDataPtr, (int)outTypeSize);
            var auxTypeData = PtrToArrayInt32(auxTypeDataPtr, (int)auxTypeSize);

            if (complete > 0)
            {
              if(inTypeSize!=0) {  inType?.AddRange(inTypeData.Select(s=>Util._DTYPE_MX_TO_NP[s]));}
                if (outTypeSize != 0) { outType?.AddRange(outTypeData.Select(s => Util._DTYPE_MX_TO_NP[s]));}
                if (auxTypeSize != 0) { auxType?.AddRange(auxTypeData.Select(s => Util._DTYPE_MX_TO_NP[s]));}
            }
        }

    

        /// <summary>
        /// Infer the shape of outputs and arguments of given known shapes of arguments
        /// </summary>
        /// <param name="argShapes"> Provide keyword arguments of known shapes.</param>
        /// <param name="inShape">List of shapes of arguments.The order is in the same order as list_arguments()</param>
        /// <param name="auxShape">List of shapes of outputs.The order is in the same order as list_outputs()</param>
        /// <param name="outShape">List of shapes of outputs.The order is in the same order as list_auxiliary()</param>
        public void InferShape(
            Dictionary<string, uint[]> argShapes,
            [Out] List<uint[]> inShape,
            [Out] List<uint[]> auxShape,
            [Out] List<uint[]> outShape)
        {
            var keys = new List<string>();
            var argIndPtr = new List<uint>();
            var argShapeData = new List<uint>();

            foreach (var arg in argShapes)
            {
                keys.Add(arg.Key);
                argIndPtr.Add((uint)argShapeData.Count);
                foreach (var i in arg.Value)
                {
                    argShapeData.Add(i);
                }
            }
            argIndPtr.Add((uint)argShapeData.Count);

            uint inShapeSize;
            IntPtr inShapeNdimPtr;
            IntPtr inShapeDataPtr;
            uint outShapeSize;
            IntPtr outShapeNdimPtr;
            IntPtr outShapeDataPtr;
            uint auxShapeSize;
            IntPtr auxShapeNdimPtr;
            IntPtr auxShapeDataPtr;
            int complete;

            Debug.Assert(NativeMethods.MXSymbolInferShape(GetHandle(), (uint)keys.Count, keys.ToArray(),
                argIndPtr.ToArray(), argShapeData.ToArray(),
                out inShapeSize, out inShapeNdimPtr, out inShapeDataPtr,
                out outShapeSize, out outShapeNdimPtr, out outShapeDataPtr,
                out auxShapeSize, out auxShapeNdimPtr, out auxShapeDataPtr,
                out complete) ==
                         0, NativeMethods.MXGetLastError());

            var inShapeNdim = PtrToArrayUint32(inShapeNdimPtr, (int)inShapeSize);
            var inShapeData = PtrToArrayUint32(inShapeDataPtr, (int)inShapeSize, inShapeNdim);

            var outShapeNdim = PtrToArrayUint32(outShapeNdimPtr, (int)outShapeSize);
            var outShapeData = PtrToArrayUint32(outShapeDataPtr, (int)outShapeSize, outShapeNdim);

            var auxShapeNdim = PtrToArrayUint32(auxShapeNdimPtr, (int)auxShapeSize);
            var auxShapeData = PtrToArrayUint32(auxShapeDataPtr, (int)auxShapeSize, auxShapeNdim);

            if (complete > 0)
            {
                if (inShapeSize != 0) { inShape?.AddRange(inShapeData); }
                if (outShapeSize != 0) { outShape?.AddRange(outShapeData); }
                if (auxShapeSize != 0) { auxShape?.AddRange(auxShapeData); }
            }
        }


        void InferExecutorArrays(
            Context context, List<NDArray> arg_arrays,
            List<NDArray> grad_arrays, List<OpReqType> grad_reqs,
            List<NDArray> aux_arrays,
            Dictionary<string, NDArray> args_map,
            Dictionary<string, NDArray> arg_grad_store,
            Dictionary<string, OpReqType> grad_req_type,
            Dictionary<string, NDArray> aux_map)
        {

            var arg_name_list = ListArguments();
            var in_shapes = new List<uint[]>();
            var aux_shapes = new List<uint[]>();
            var out_shapes = new List<uint[]>();
            var arg_shapes = new Dictionary<string, uint[]>();

            foreach (var arg_name in arg_name_list)
            {
                if (args_map.ContainsKey(arg_name))
                {
                    arg_shapes[arg_name] = args_map[arg_name].GetShape();
                }
            }


            InferShape(arg_shapes, in_shapes, aux_shapes, out_shapes);

            for (int i = 0; i < in_shapes.Count; ++i)
            {
                var shape = in_shapes[i];
                var arg_name = arg_name_list[i];

                if (args_map.ContainsKey(arg_name))
                {
                    arg_arrays.Add(args_map[arg_name]);
                }
                else
                {
                    var temp = new NDArray(shape, context, false);
                    arg_arrays.Add(temp);
                    NDArray.SampleGaussian(0, 1, temp);
                }

                if (arg_grad_store.ContainsKey(arg_name))
                {
                    grad_arrays.Add(arg_grad_store[arg_name]);
                }
                else
                {
                    grad_arrays.Add(new NDArray(shape, context, false));
                }
                if (grad_req_type.ContainsKey(arg_name))
                {
                    grad_reqs.Add(grad_req_type[arg_name]);
                }
                else
                {
                    grad_reqs.Add(csharp.OpReqType.KWriteTo);
                }
            }

            var aux_name_list = ListAuxiliaryStates();
            for (int i = 0; i < aux_shapes.Count; ++i)
            {
                var shape = aux_shapes[i];
                var aux_name = aux_name_list[i];
                if (aux_map.ContainsKey(aux_name))
                {
                    aux_arrays.Add(aux_map[aux_name]);
                }
                else
                {
                    var temp = new NDArray(shape, context, false);
                    aux_arrays.Add(temp);
                    csharp.NDArray.SampleGaussian(0, 1, temp);
                }
            }
        }

        public void InferArgsMap(
            Context context, Dictionary<string, NDArray> argsMap,
            Dictionary<string, NDArray> knownArgs)
        {
            var argNameList = ListArguments();
            var inShapes = new List<uint[]>();
            var auxShapes = new List<uint[]>();
            var outShapes = new List<uint[]>();
            var argShapes = new Dictionary<string, uint[]>();

            foreach (var argName in argNameList)
            {
                if (knownArgs.ContainsKey(argName))
                {
                    argShapes[argName] = knownArgs[argName].GetShape();
                }
            }

            InferShape(argShapes, inShapes, auxShapes, outShapes);

            for (var i = 0; i < inShapes.Count; ++i)
            {
                var shape = inShapes[i];
                var argName = argNameList[i];
                if (knownArgs.ContainsKey(argName))
                {
                    argsMap[argName] = knownArgs[argName];
                }
                else
                {
                    argsMap[argName] = new NDArray(shape, context, false);
                    NDArray.SampleGaussian(0, 1, argsMap[argName]);
                }
            }
        }

        public Executor SimpleBind(
            Context context, Dictionary<string, NDArray> args_map,
            Dictionary<string, NDArray> arg_grad_store =null,
            Dictionary<string, OpReqType> grad_req_type = null,
            Dictionary<string, NDArray> aux_map = null)
        {
            if (arg_grad_store == null)
            {
                arg_grad_store = new Dictionary<string, NDArray>();
            }
            if (grad_req_type == null)
            {
                grad_req_type = new Dictionary<string, OpReqType>();
            }
            if (aux_map == null)
            {
                aux_map = new Dictionary<string, NDArray>();
            }

            List<NDArray> arg_arrays = new List<NDArray>();
            List<NDArray> grad_arrays = new List<NDArray>();
            List<OpReqType> grad_reqs = new List<OpReqType>();
            List<NDArray> aux_arrays = new List<NDArray>();

            InferExecutorArrays(context, arg_arrays, grad_arrays, grad_reqs,
                                aux_arrays, args_map, arg_grad_store, grad_req_type,
                                aux_map);

            return new Executor(this, context, arg_arrays, grad_arrays, grad_reqs,
                                aux_arrays);
        }

        public Executor Bind(Context context,
            List<NDArray> arg_arrays,
            List<NDArray> grad_arrays,
            List<OpReqType> grad_reqs,
            List<NDArray> aux_arrays,
            Dictionary<string, Context> group_to_ctx = null,
            Executor shared_exec = null)
        {
            return new Executor(this, context, arg_arrays, grad_arrays, grad_reqs,
                                aux_arrays, group_to_ctx, shared_exec);
        }

        public Executor Bind(Context context,
            List<NDArray> arg_arrays,
            Dictionary<string, NDArray> grad_dict,
            Dictionary<string, OpReqType> grad_reqs,
            List<NDArray> aux_arrays,
            Dictionary<string, Context> group_to_ctx = null,
            Executor shared_exec = null)
        {
            var listed_arguments = this.ListArguments();
            var grad_arrays = this._get_ndarray_inputs("args_grad", grad_dict, listed_arguments, true);
           
            return new Executor(this, context, arg_arrays, grad_arrays,
                grad_reqs.Select(s => s.Value).ToList(), aux_arrays, group_to_ctx, shared_exec);
        }

        private List<NDArray> _get_ndarray_inputs(string arg_key, Dictionary<string, NDArray> args, IList<string> arg_names, bool allow_missing)
        {
            List<NDArray> arg_arrays = new List<NDArray>();
            foreach (var name in arg_names)
            {
                if (args.ContainsKey(name))
                {
                    arg_arrays.Add(args[name]);
                }
                else
                {
                    if (allow_missing)
                    {
                        arg_arrays.Add(null);
                    }
                    else
                    {
                        throw new Exception($"Must specify all the arguments in {arg_key}" );
                    }
                }
            }
            return arg_arrays;
        }

        /// <summary>
        /// Get all attributes from the symbol
        /// </summary>
        /// <param name="recursive">
        /// Default `False`. When `recursive` is `True`, list recursively all the
        /// attributes in the descendents. The attribute names are pre-pended with
        /// the symbol names to avoid conflicts. If `False`, then only attributes
        /// that belongs to this symbol is returned, and the attribute names will
        /// **not** be pre-pended with the symbol name.
        /// </param>
        public Dictionary<string, string> list_attr(bool recursive = false)
        {
            uint out_size;
            IntPtr out_ptr;
            if (recursive)
            {
                NativeMethods.MXSymbolListAttr(GetHandle(), out out_size, out out_ptr);
            }
            else
            {
                NativeMethods.MXSymbolListAttrShallow(GetHandle(), out out_size, out out_ptr);
            }
            IntPtr[] out_ptr_array = new IntPtr[out_size*2];

            Dictionary<string, string> attr = new Dictionary<string, string>();
            for (int i = 0; i < out_size; i++)
            {
                attr.Add(Marshal.PtrToStringAnsi(out_ptr_array[i*2]), Marshal.PtrToStringAnsi(out_ptr_array[i*2] + 1));
            }

            return attr;

        }

        public SymbolHandle GetHandle()
        {
            return _blobPtr.Handle;
        }


   
    }
}
