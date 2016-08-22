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
                NativeMethods.MXSymbolGetOutput(GetHandle(), (uint) index, out @out);
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
            var handle_list = new List<SymbolHandle>();
            foreach (var symbol in symbols)
            {
                handle_list.Add(symbol.GetHandle());
            }
            SymbolHandle @out;

            NativeMethods.MXSymbolCreateGroup((uint) handle_list.Count, handle_list.ToArray(), out @out);
            return new Symbol(@out);
        }

        public Symbol Load(string file_name)
        {
            SymbolHandle handle;
            Debug.Assert(NativeMethods.MXSymbolCreateFromFile(file_name, out handle) == 0);
            return new Symbol(handle);
        }

        public Symbol LoadJSON(string json_str)
        {
            SymbolHandle handle;
            Debug.Assert(NativeMethods.MXSymbolCreateFromJSON(json_str, out handle) == 0);
            return new Symbol(handle);
        }

        private void Save(string file_name)
        {
            Debug.Assert(NativeMethods.MXSymbolSaveToFile(GetHandle(), file_name) == 0);
        }

        public string ToJSON()
        {
            IntPtr out_json;
            Debug.Assert(NativeMethods.MXSymbolSaveToJSON(GetHandle(), out out_json) == 0);
            return Marshal.PtrToStringAnsi(out_json);
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
                Marshal.Copy(sarrPtr, sarr, 0, (int) size);
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
                Marshal.Copy(sarrPtr, sarr, 0, (int) size);
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
                Marshal.Copy(sarrPtr, sarr, 0, (int) size);
            }
            for (var i = 0; i < size; i++)
            {
                ret.Add(Marshal.PtrToStringAnsi(sarr[i]));
            }
            return ret;
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
            if (dataSize== 0)
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
                ret.Add(data.Select(s => (uint) s).ToArray());
            }

            return ret;
        }
        public void InferShape(
            Dictionary<string, uint[]> argShapes,
            List<uint[]> inShape,
            List<uint[]> auxShape,
            List<uint[]> outShape)
        {
            var keys = new List<string>();
            var argIndPtr = new List<uint>();
            var argShapeData = new List<uint>();

            foreach (var arg in argShapes)
            {
                keys.Add(arg.Key);
                argIndPtr.Add((uint) argShapeData.Count);
                foreach (var i in arg.Value)
                {
                    argShapeData.Add(i);
                }
            }
            argIndPtr.Add((uint) argShapeData.Count);

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

            Debug.Assert(NativeMethods.MXSymbolInferShape(GetHandle(), (uint) keys.Count, keys.ToArray(),
                argIndPtr.ToArray(), argShapeData.ToArray(),
                out inShapeSize, out inShapeNdimPtr, out inShapeDataPtr,
                out outShapeSize, out outShapeNdimPtr, out outShapeDataPtr,
                out auxShapeSize, out auxShapeNdimPtr, out auxShapeDataPtr,
                out complete) ==
                         0, NativeMethods.MXGetLastError());

            var inShapeNdim = PtrToArrayUint32(inShapeNdimPtr, (int)inShapeSize);
            var inShapeData = PtrToArrayUint32(inShapeDataPtr, (int)inShapeSize , inShapeNdim);

            var outShapeNdim = PtrToArrayUint32(outShapeNdimPtr, (int)outShapeSize);
            var outShapeData = PtrToArrayUint32(outShapeDataPtr, (int)outShapeSize, outShapeNdim);

            var auxShapeNdim = PtrToArrayUint32(auxShapeNdimPtr, (int)auxShapeSize);
            var auxShapeData = PtrToArrayUint32(auxShapeDataPtr, (int)auxShapeSize, auxShapeNdim);

            if (complete > 0)
            {
                if (inShapeData != null) { inShape.AddRange(inShapeData);}
                if (outShapeData != null) { outShape.AddRange(outShapeData);}
                if (auxShapeData != null) { auxShape.AddRange(auxShapeData);}
            }
        }

  

        public void InferArgsMap(
            Context context, Dictionary<string, NDArray> args_map,
            Dictionary<string, NDArray> known_args)
        {
            var arg_name_list = ListArguments();
            var in_shapes = new List<uint[]>();
            var aux_shapes = new List<uint[]>();
            var out_shapes = new List<uint[]>();
            var arg_shapes = new Dictionary<string, uint[]>();

            foreach (var arg_name in arg_name_list)
            {
                if (known_args.ContainsKey(arg_name))
                    arg_shapes[arg_name] = known_args[arg_name].GetShape();
            }

            InferShape(arg_shapes, in_shapes, aux_shapes, out_shapes);

            for (var i = 0; i < in_shapes.Count; ++i)
            {
                var shape = in_shapes[i];
                var arg_name = arg_name_list[i];
                if (known_args.ContainsKey(arg_name))
                {
                    args_map[arg_name] = known_args[arg_name];
                }
                else
                {
                    args_map[arg_name] = new NDArray(shape, context, false);
                    //   NDArray,SampleGaussian(0, 1, args_map[arg_name]);
                }
            }
        }

        public SymbolHandle GetHandle()
        {
            return _blobPtr.Handle;
        }
    }
}
