using System;
using System.Collections;
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
            this.Handle = handle;
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

    public partial class Symbol
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
                NativeMethods.MXSymbolGetOutput(get_handle(), (uint)index, out @out);
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
                handleList.Add(symbol.get_handle());
            }
            SymbolHandle @out;

            NativeMethods.MXSymbolCreateGroup((uint)handleList.Count, handleList.ToArray(), out @out);
            return new Symbol(@out);
        }

        public static Symbol Load(string fileName)
        {
            SymbolHandle handle;
            Util.CallCheck(NativeMethods.MXSymbolCreateFromFile(fileName, out handle));
            return new Symbol(handle);
        }

        public Symbol LoadJson(string jsonStr)
        {
            SymbolHandle handle;
            Util.CallCheck(NativeMethods.MXSymbolCreateFromJSON(jsonStr, out handle));
            return new Symbol(handle);
        }

        public void Save(string fileName)
        {
            Util.CallCheck(NativeMethods.MXSymbolSaveToFile(get_handle(), fileName));
        }

        public string ToJson()
        {
            IntPtr outJson;
            Util.CallCheck(NativeMethods.MXSymbolSaveToJSON(get_handle(), out outJson));
            return Marshal.PtrToStringAnsi(outJson);
        }

        public Symbol GetInternals()
        {
            SymbolHandle handle;
            Util.CallCheck(NativeMethods.MXSymbolGetInternals(get_handle(), out handle));
            return new Symbol(handle);
        }

        public Symbol Copy()
        {
            SymbolHandle handle;
            Util.CallCheck(NativeMethods.MXSymbolCopy(get_handle(), out handle));
            return new Symbol(handle);
        }

        public IList<string> ListArguments()
        {
            var ret = new List<string>();
            uint size;
            IntPtr sarrPtr;

            NativeMethods.MXSymbolListArguments(get_handle(), out size, out sarrPtr);
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

            NativeMethods.MXSymbolListOutputs(get_handle(), out size, out sarrPtr);
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

            NativeMethods.MXSymbolListAuxiliaryStates(get_handle(), out size, out sarrPtr);
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
            if (size ==0)
            {
                return null;
            }
            int[] array = new int[size];
            Marshal.Copy(ptr, array, 0, size);
            return array;
        }

        private static uint[] PtrToArrayUint32(IntPtr ptr, int size)
        {
            if (size==0)
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
            if (dataSize==0)
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
            InferShape(argShapes.ToDictionary(x => x.Key, y => (uint[])y.Value), inShape, outShape, auxShape);
        }


        public void InferType(
         Dictionary<string, Type> inputTypes,
         [Out] List<Type> inType,
         [Out] List<Type> auxType,
         [Out] List<Type> outType)
        {
            var keys = new List<string>();
            var argTypeData = new List<int>();


            foreach (var arg in inputTypes)
            {
                keys.Add(arg.Key);
                argTypeData.Add(Util.DtypeNpToMx[arg.Value]);
            }


            uint inTypeSize;
            IntPtr inTypeDataPtr;

            uint outTypeSize;
            IntPtr outTypeDataPtr;

            uint auxTypeSize;
            IntPtr auxTypeDataPtr;

            int complete;

            Util.CallCheck(NativeMethods.MXSymbolInferType(get_handle(), (uint)keys.Count, keys.ToArray(),
                argTypeData.ToArray(),
                out inTypeSize, out inTypeDataPtr,
                out outTypeSize, out outTypeDataPtr,
                out auxTypeSize, out auxTypeDataPtr,
                out complete));

            var inTypeData = PtrToArrayInt32(inTypeDataPtr, (int)inTypeSize);
            var outTypeData = PtrToArrayInt32(outTypeDataPtr, (int)outTypeSize);
            var auxTypeData = PtrToArrayInt32(auxTypeDataPtr, (int)auxTypeSize);

            if (complete > 0)
            {
                if (inTypeSize != 0) { inType?.AddRange(inTypeData.Select(s => Util.DtypeMxToNp[s])); }
                if (outTypeSize != 0) { outType?.AddRange(outTypeData.Select(s => Util.DtypeMxToNp[s])); }
                if (auxTypeSize != 0) { auxType?.AddRange(auxTypeData.Select(s => Util.DtypeMxToNp[s])); }
            }
        }


        /// <summary>
        /// Infer the shape of outputs and arguments of given known shapes of arguments
        /// </summary>
        /// <param name="argShapes"> Provide keyword arguments of known shapes.</param>
        /// <param name="inShape">List of shapes of arguments.The order is in the same order as list_arguments()</param>
        /// <param name="outShape">List of shapes of outputs.The order is in the same order as list_outputs()</param>
        /// <param name="auxShape">List of shapes of outputs.The order is in the same order as list_auxiliary()</param>
        public void InferShape(Dictionary<string, uint[]> argShapes, [Out] List<uint[]> inShape, [Out] List<uint[]> outShape, [Out] List<uint[]> auxShape)
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

            Util.CallCheck(NativeMethods.MXSymbolInferShape(get_handle(), (uint)keys.Count, keys.ToArray(),
                argIndPtr.ToArray(), argShapeData.ToArray(),
                out inShapeSize, out inShapeNdimPtr, out inShapeDataPtr,
                out outShapeSize, out outShapeNdimPtr, out outShapeDataPtr,
                out auxShapeSize, out auxShapeNdimPtr, out auxShapeDataPtr,
                out complete));

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
            Context context, IList<NdArray> argArrays,
            IList<NdArray> gradArrays, IList<OpReqType> gradReqs,
            IList<NdArray> auxArrays,
            Dictionary<string, NdArray> argsMap,
            Dictionary<string, NdArray> argGradStore,
            Dictionary<string, OpReqType> gradReqType,
            Dictionary<string, NdArray> auxMap)
        {

            var argNameList = ListArguments();
            var inShapes = new List<uint[]>();
            var auxShapes = new List<uint[]>();
            var outShapes = new List<uint[]>();
            var argShapes = new Dictionary<string, uint[]>();

            foreach (var argName in argNameList)
            {
                if (argsMap.ContainsKey(argName))
                {
                    argShapes[argName] = argsMap[argName].GetShape();
                }
            }


            InferShape(argShapes, inShapes, outShapes, auxShapes);

            for (int i = 0; i < inShapes.Count; ++i)
            {
                var shape = inShapes[i];
                var argName = argNameList[i];

                if (argsMap.ContainsKey(argName))
                {
                    argArrays.Add(argsMap[argName]);
                }
                else
                {
                    var temp = new NdArray(shape, context, false);
                    argArrays.Add(temp);
                    NdArray.SampleGaussian(0, 1, temp);
                }

                if (argGradStore.ContainsKey(argName))
                {
                    gradArrays.Add(argGradStore[argName]);
                }
                else
                {
                    gradArrays.Add(new NdArray(shape, context, false));
                }
                if (gradReqType.ContainsKey(argName))
                {
                    gradReqs.Add(gradReqType[argName]);
                }
                else
                {
                    gradReqs.Add(csharp.OpReqType.KWriteTo);
                }
            }

            var auxNameList = ListAuxiliaryStates();
            for (int i = 0; i < auxShapes.Count; ++i)
            {
                var shape = auxShapes[i];
                var auxName = auxNameList[i];
                if (auxMap.ContainsKey(auxName))
                {
                    auxArrays.Add(auxMap[auxName]);
                }
                else
                {
                    var temp = new NdArray(shape, context, false);
                    auxArrays.Add(temp);
                    csharp.NdArray.SampleGaussian(0, 1, temp);
                }
            }
        }

        public void InferArgsMap(
            Context context, Dictionary<string, NdArray> argsMap,
            Dictionary<string, NdArray> knownArgs)
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

            InferShape(argShapes, inShapes, outShapes, auxShapes);

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
                    argsMap[argName] = new NdArray(shape, context, false);
                    NdArray.SampleGaussian(0, 1, argsMap[argName]);
                }
            }
        }

        public Executor SimpleBind(
            Context context,
            Dictionary<string, uint[]> inputShapes,
            OpReqType gradReq,
            Dictionary<string, Type> typeDict = null,
            Dictionary<string, Context> group2Ctx = null
            )
        {

           var  listArguments = ListArguments();


            if (typeDict == null)
            {
                typeDict = listArguments.ToDictionary(k => k, v => typeof(float));
            }

            var argShapes = new List<uint[]>();
            var auxShapes = new List<uint[]>();
            var outShapes = new List<uint[]>();
            InferShape(inputShapes, argShapes, outShapes, auxShapes);


            var argTypes = new List<Type>();
            var auxTypes = new List<Type>();
            var outTypes = new List<Type>();

            InferType(typeDict, argTypes, auxTypes, outTypes);

            if (argShapes.Count == 0|| argTypes.Count == 0)
            {
                throw new Exception("Input node is not complete");
            }

            List<Context> argCtx;
            List<Context> auxCtx;
            if (group2Ctx != null)
            {
                var listattr = ListAttr(true);
                var attrDict = listattr.Where(w => w.Key.EndsWith("ctx_group"))
                    .ToDictionary(k => k.Key, v => group2Ctx.GetValueOrDefault(v.Value, context));
                argCtx = listArguments
                    .Select(name => attrDict.GetValueOrDefault(name + "_ctx_group", context)).ToList();
                auxCtx = ListAuxiliaryStates()
                    .Select(name => attrDict.GetValueOrDefault(name + "_ctx_group", context)).ToList();
            }
            else
            {
                argCtx = Enumerable.Range(0, argShapes.Count).Select(s => context).ToList();
                auxCtx = Enumerable.Range(0, auxShapes.Count).Select(s => context).ToList();
            }

            //alloc space
            var argNdarrays = argTypes
                .Zip(argCtx, argShapes,
                    (dtype, dev, shape) =>
                        NdArray.Zeros(new Shape(shape), dev, dtype))
                .ToList();
            Dictionary<string, NdArray> gradNdarrays = new Dictionary<string, NdArray>();
            if (gradReq != OpReqType.KNullOp)
            {
             
                for (int i = 0; i < listArguments.Count; i++)
                {
                    var name = listArguments[i];
                    var shape = argShapes[i];
                    var dev = argCtx[i];
                    var dtype = argTypes[i];

                    gradNdarrays[name] = NdArray.Zeros(new Shape(shape), dev, dtype: dtype);
                }
            }

            var auxNdarrays = auxTypes
                .Zip(auxCtx, auxShapes,
                    (dtype, dev, shape) =>
                        NdArray.Zeros(new Shape(shape), dev, dtype))
                .ToList();

            var executor = Bind(context,
                argNdarrays,
                gradNdarrays, 
                gradReq,
                auxNdarrays,
                groupToCtx: group2Ctx);

            return executor;
        }

        public Executor Bind(Context context,
            IList<NdArray> argArrays,
            IList<NdArray> gradArrays,
            IList<OpReqType> gradReqs,
            IList<NdArray> auxArrays,
            Dictionary<string, Context> groupToCtx = null,
            Executor sharedExec = null)
        {
            return new Executor(this, context, argArrays, gradArrays, gradReqs,
                                auxArrays, groupToCtx, sharedExec);
        }

        public Executor Bind(Context context,
            IList<NdArray> argArrays,
            Dictionary<string, NdArray> gradDict,
            OpReqType gradReq,
            IList<NdArray> auxArrays,
            Dictionary<string, Context> groupToCtx = null,
            Executor sharedExec = null)
        {
            var gradReqs = ListArguments().ToDictionary(k => k, v => gradReq);
            return Bind(context, argArrays, gradDict, gradReqs, auxArrays, groupToCtx, sharedExec);
        }

        public Executor Bind(Context context,
            IList<NdArray> argArrays,
            Dictionary<string, NdArray> gradDict,
            Dictionary<string, OpReqType> gradReqs,
            IList<NdArray> auxArrays,
            Dictionary<string, Context> groupToCtx = null,
            Executor sharedExec = null)
        {
            var listedArguments = this.ListArguments();
            var gradArrays = this.GetNDarrayInputs("args_grad", gradDict, listedArguments, true);

            return new Executor(this, context, argArrays, gradArrays,
                gradReqs.Select(s => s.Value).ToList(), auxArrays, groupToCtx, sharedExec);
        }

        private List<NdArray> GetNDarrayInputs(string argKey, Dictionary<string, NdArray> args, IList<string> argNames, bool allowMissing)
        {
            List<NdArray> argArrays = new List<NdArray>();
            foreach (var name in argNames)
            {
                if (args.ContainsKey(name))
                {
                    argArrays.Add(args[name]);
                }
                else
                {
                    if (allowMissing)
                    {
                        argArrays.Add(null);
                    }
                    else
                    {
                        throw new Exception($"Must specify all the arguments in {argKey}");
                    }
                }
            }
            return argArrays;
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
        public Dictionary<string, string> ListAttr(bool recursive = false)
        {
            uint outSize;
            IntPtr outPtr;
            if (recursive)
            {
                NativeMethods.MXSymbolListAttr(get_handle(), out outSize, out outPtr);
            }
            else
            {
                NativeMethods.MXSymbolListAttrShallow(get_handle(), out outSize, out outPtr);
            }
            IntPtr[] outPtrArray = new IntPtr[outSize * 2];

            Dictionary<string, string> attr = new Dictionary<string, string>();
            for (int i = 0; i < outSize; i++)
            {
                attr.Add(Marshal.PtrToStringAnsi(outPtrArray[i * 2]), Marshal.PtrToStringAnsi(outPtrArray[i * 2] + 1));
            }

            return attr;

        }
        [DebuggerHidden]
        public SymbolHandle get_handle()
        {
            return _blobPtr.Handle;
        }



    }
}
