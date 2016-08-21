using System;
using System.Collections.Generic;
using System.Diagnostics;
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

        public void InferShape(
            Dictionary<string, uint[]> arg_shapes,
            List<uint[]> in_shape,
            List<uint[]> aux_shape,
            List<uint[]> out_shape)
        {
            var keys = new List<string>();
            var arg_ind_ptr = new List<uint>();
            var arg_shape_data = new List<uint>();

            foreach (var arg in arg_shapes)
            {
                keys.Add(arg.Key);
                arg_ind_ptr.Add((uint) arg_shape_data.Count);
                foreach (var i in arg.Value)
                {
                    arg_shape_data.Add(i);
                }
            }
            arg_ind_ptr.Add((uint) arg_shape_data.Count);

            uint in_shape_size;
            IntPtr in_shape_ndim;
            IntPtr in_shape_data;
            uint out_shape_size;
            IntPtr out_shape_ndim_ptr;
            IntPtr out_shape_data_ptr;
            uint aux_shape_size;
            IntPtr aux_shape_ndim_ptr;
            IntPtr aux_shape_data_ptr;
            int complete;

            Debug.Assert(NativeMethods.MXSymbolInferShape(GetHandle(), (uint) keys.Count, keys.ToArray(),
                arg_ind_ptr.ToArray(), arg_shape_data.ToArray(),
                out in_shape_size, out in_shape_ndim, out in_shape_data,
                out out_shape_size, out out_shape_ndim_ptr, out out_shape_data_ptr,
                out aux_shape_size, out aux_shape_ndim_ptr, out aux_shape_data_ptr,
                out complete) ==
                         0, NativeMethods.MXGetLastError());
            var message = NativeMethods.MXGetLastError();

            var m = Marshal.PtrToStructure(out_shape_ndim_ptr, typeof(uint[]));


            //if (complete>0) {
            //  for (uint i = 0; i<in_shape_size; ++i) {
            //    in_shape.Add(std::vector<mx_uint>());

            //    for (mx_uint j = 0; j<in_shape_ndim[i]; ++j) {
            //      (* in_shape)[i].push_back(in_shape_data[i][j]);
            //    }
            //  }
            //  for (uint i = 0; i<aux_shape_size; ++i) {
            //    aux_shape->push_back(std::vector<mx_uint>());
            //    for (mx_uint j = 0; j<aux_shape_ndim[i]; ++j) {
            //      (* aux_shape)[i].push_back(aux_shape_data[i][j]);
            //    }
            //  }
            //  for (uint i = 0; i<out_shape_size; ++i) {
            //    out_shape->push_back(std::vector<mx_uint>());
            //    for (mx_uint j = 0; j<out_shape_ndim[i]; ++j) {
            //      (* out_shape)[i].push_back(out_shape_data[i][j]);
            //    }
            //}
            // }
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
