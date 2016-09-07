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
            this.handle = handle;
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
            NativeMethods.MXSymbolFree(handle);
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
        public SymbolHandle handle { get; } = IntPtr.Zero;
    }

    public partial class Symbol
    {
        private readonly SymBlob _blob_ptr;

        public Symbol(string name)
        {
            SymbolHandle symbol_handle;
            NativeMethods.MXSymbolCreateVariable(name, out symbol_handle);
            _blob_ptr = new SymBlob(symbol_handle);
        }

        public Symbol(SymbolHandle symbol_handle)
        {
            _blob_ptr = new SymBlob(symbol_handle);
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
            var handle_list = new List<SymbolHandle>();
            foreach (var symbol in symbols)
            {
                handle_list.Add(symbol.GetHandle());
            }
            SymbolHandle @out;

            NativeMethods.MXSymbolCreateGroup((uint)handle_list.Count, handle_list.ToArray(), out @out);
            return new Symbol(@out);
        }

        public Symbol Load(string file_name)
        {
            SymbolHandle handle;
            Util.CallCheck(NativeMethods.MXSymbolCreateFromFile(file_name, out handle));
            return new Symbol(handle);
        }

        public Symbol LoadJson(string json_str)
        {
            SymbolHandle handle;
            Util.CallCheck(NativeMethods.MXSymbolCreateFromJSON(json_str, out handle));
            return new Symbol(handle);
        }

        private void Save(string file_name)
        {
            Util.CallCheck(NativeMethods.MXSymbolSaveToFile(GetHandle(), file_name));
        }

        public string ToJson()
        {
            IntPtr out_json;
            Util.CallCheck(NativeMethods.MXSymbolSaveToJSON(GetHandle(), out out_json));
            return Marshal.PtrToStringAnsi(out_json);
        }

        public Symbol GetInternals()
        {
            SymbolHandle handle;
            Util.CallCheck(NativeMethods.MXSymbolGetInternals(GetHandle(), out handle));
            return new Symbol(handle);
        }

        public Symbol Copy()
        {
            SymbolHandle handle;
            Util.CallCheck(NativeMethods.MXSymbolCopy(GetHandle(), out handle));
            return new Symbol(handle);
        }

        public IList<string> ListArguments()
        {
            var ret = new List<string>();
            uint size;
            IntPtr sarr_ptr;

            NativeMethods.MXSymbolListArguments(GetHandle(), out size, out sarr_ptr);
            var sarr = new IntPtr[size];
            if (size > 0)
            {
                Marshal.Copy(sarr_ptr, sarr, 0, (int)size);
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
            IntPtr sarr_ptr;

            NativeMethods.MXSymbolListOutputs(GetHandle(), out size, out sarr_ptr);
            var sarr = new IntPtr[size];
            if (size > 0)
            {
                Marshal.Copy(sarr_ptr, sarr, 0, (int)size);
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
            IntPtr sarr_ptr;

            NativeMethods.MXSymbolListAuxiliaryStates(GetHandle(), out size, out sarr_ptr);
            var sarr = new IntPtr[size];
            if (size > 0)
            {
                Marshal.Copy(sarr_ptr, sarr, 0, (int)size);
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
        private static List<uint[]> PtrToArrayUint32(IntPtr data_ptr, int data_size, uint[] shape_ndim)
        {
            if (data_size==0)
            {
                return null;
            }
            IntPtr[] ptr_array = new IntPtr[data_size];
            Marshal.Copy(data_ptr, ptr_array, 0, data_size);
            List<uint[]> ret = new List<uint[]>();

            for (int i = 0; i < data_size; i++)
            {
                int[] data = new int[(int)shape_ndim[i]];
                Marshal.Copy(ptr_array[i], data, 0, (int)shape_ndim[i]);
                ret.Add(data.Select(s => (uint)s).ToArray());
            }

            return ret;
        }
        /// <summary>
        /// Infer the shape of outputs and arguments of given known shapes of arguments
        /// </summary>
        /// <param name="arg_shapes"> Provide keyword arguments of known shapes.</param>
        /// <param name="in_shape">List of shapes of arguments.The order is in the same order as list_arguments()</param>
        /// <param name="aux_shape">List of shapes of outputs.The order is in the same order as list_outputs()</param>
        /// <param name="out_shape">List of shapes of outputs.The order is in the same order as list_auxiliary()</param>
        public void InferShape(Dictionary<string, Shape> arg_shapes, List<uint[]> in_shape, List<uint[]> aux_shape, List<uint[]> out_shape)
        {
            InferShape(arg_shapes.ToDictionary(x => x.Key, y => (uint[])y.Value), in_shape, aux_shape, out_shape);
        }


        public void InferType(
         Dictionary<string, Type> input_types,
         [Out] List<Type> in_type,
         [Out] List<Type> aux_type,
         [Out] List<Type> out_type)
        {
            var keys = new List<string>();
            var arg_type_data = new List<int>();


            foreach (var arg in input_types)
            {
                keys.Add(arg.Key);
                arg_type_data.Add(Util.DtypeNpToMX[arg.Value]);
            }


            uint in_type_size;
            IntPtr in_type_data_ptr;

            uint out_type_size;
            IntPtr out_type_data_ptr;

            uint aux_type_size;
            IntPtr aux_type_data_ptr;

            int complete;

            Util.CallCheck(NativeMethods.MXSymbolInferType(GetHandle(), (uint)keys.Count, keys.ToArray(),
                arg_type_data.ToArray(),
                out in_type_size, out in_type_data_ptr,
                out out_type_size, out out_type_data_ptr,
                out aux_type_size, out aux_type_data_ptr,
                out complete));

            var in_type_data = PtrToArrayInt32(in_type_data_ptr, (int)in_type_size);
            var out_type_data = PtrToArrayInt32(out_type_data_ptr, (int)out_type_size);
            var aux_type_data = PtrToArrayInt32(aux_type_data_ptr, (int)aux_type_size);

            if (complete > 0)
            {
                if (in_type_size != 0) { in_type?.AddRange(in_type_data.Select(s => Util.DtypeMXToNp[s])); }
                if (out_type_size != 0) { out_type?.AddRange(out_type_data.Select(s => Util.DtypeMXToNp[s])); }
                if (aux_type_size != 0) { aux_type?.AddRange(aux_type_data.Select(s => Util.DtypeMXToNp[s])); }
            }
        }



        /// <summary>
        /// Infer the shape of outputs and arguments of given known shapes of arguments
        /// </summary>
        /// <param name="arg_shapes"> Provide keyword arguments of known shapes.</param>
        /// <param name="in_shape">List of shapes of arguments.The order is in the same order as list_arguments()</param>
        /// <param name="aux_shape">List of shapes of outputs.The order is in the same order as list_outputs()</param>
        /// <param name="out_shape">List of shapes of outputs.The order is in the same order as list_auxiliary()</param>
        public void InferShape(
            Dictionary<string, uint[]> arg_shapes,
            [Out] List<uint[]> in_shape,
            [Out] List<uint[]> aux_shape,
            [Out] List<uint[]> out_shape)
        {
            var keys = new List<string>();
            var arg_ind_ptr = new List<uint>();
            var arg_shape_data = new List<uint>();

            foreach (var arg in arg_shapes)
            {
                keys.Add(arg.Key);
                arg_ind_ptr.Add((uint)arg_shape_data.Count);
                foreach (var i in arg.Value)
                {
                    arg_shape_data.Add(i);
                }
            }
            arg_ind_ptr.Add((uint)arg_shape_data.Count);

            uint in_shape_size;
            IntPtr in_shape_ndim_ptr;
            IntPtr in_shape_data_ptr;
            uint out_shape_size;
            IntPtr out_shape_ndim_ptr;
            IntPtr out_shape_data_ptr;
            uint aux_shape_size;
            IntPtr aux_shape_ndim_ptr;
            IntPtr aux_shape_data_ptr;
            int complete;

            Util.CallCheck(NativeMethods.MXSymbolInferShape(GetHandle(), (uint)keys.Count, keys.ToArray(),
                arg_ind_ptr.ToArray(), arg_shape_data.ToArray(),
                out in_shape_size, out in_shape_ndim_ptr, out in_shape_data_ptr,
                out out_shape_size, out out_shape_ndim_ptr, out out_shape_data_ptr,
                out aux_shape_size, out aux_shape_ndim_ptr, out aux_shape_data_ptr,
                out complete));

            var in_shape_ndim = PtrToArrayUint32(in_shape_ndim_ptr, (int)in_shape_size);
            var in_shape_data = PtrToArrayUint32(in_shape_data_ptr, (int)in_shape_size, in_shape_ndim);

            var out_shape_ndim = PtrToArrayUint32(out_shape_ndim_ptr, (int)out_shape_size);
            var out_shape_data = PtrToArrayUint32(out_shape_data_ptr, (int)out_shape_size, out_shape_ndim);

            var aux_shape_ndim = PtrToArrayUint32(aux_shape_ndim_ptr, (int)aux_shape_size);
            var aux_shape_data = PtrToArrayUint32(aux_shape_data_ptr, (int)aux_shape_size, aux_shape_ndim);

            if (complete > 0)
            {
                if (in_shape_size != 0) { in_shape?.AddRange(in_shape_data); }
                if (out_shape_size != 0) { out_shape?.AddRange(out_shape_data); }
                if (aux_shape_size != 0) { aux_shape?.AddRange(aux_shape_data); }
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
                    arg_shapes[arg_name] = args_map[arg_name].Get_shape();
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
                    NDArray.Sample_gaussian(0, 1, temp);
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
                    csharp.NDArray.Sample_gaussian(0, 1, temp);
                }
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
                {
                    arg_shapes[arg_name] = known_args[arg_name].Get_shape();
                }
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
                    NDArray.Sample_gaussian(0, 1, args_map[arg_name]);
                }
            }
        }

        public Executor SimpleBind(
            Context context,
            Dictionary<string, uint[]> input_shapes,
            OpReqType grad_req,
            Dictionary<string, Type> type_dict = null,
            Dictionary<string, Context> group2ctx = null
            )
        {

           var  list_arguments = ListArguments();


            if (type_dict == null)
            {
                type_dict = list_arguments.ToDictionary(k => k, v => typeof(float));
            }

            var arg_shapes = new List<uint[]>();
            var aux_shapes = new List<uint[]>();
            var out_shapes = new List<uint[]>();
            InferShape(input_shapes, arg_shapes, aux_shapes, out_shapes);


            var arg_types = new List<Type>();
            var aux_types = new List<Type>();
            var out_types = new List<Type>();

            InferType(type_dict, arg_types, aux_types, out_types);

            if (arg_shapes.Count == 0|| arg_types.Count == 0)
            {
                throw new Exception("Input node is not complete");
            }

            List<Context> arg_ctx;
            List<Context> aux_ctx;
            if (group2ctx != null)
            {
                var listattr = list_attr(true);
                var attr_dict = listattr.Where(w => w.Key.EndsWith("ctx_group"))
                    .ToDictionary(k => k.Key, v => group2ctx.GetValueOrDefault(v.Value, context));
                arg_ctx = list_arguments
                    .Select(name => attr_dict.GetValueOrDefault(name + "_ctx_group", context)).ToList();
                aux_ctx = ListAuxiliaryStates()
                    .Select(name => attr_dict.GetValueOrDefault(name + "_ctx_group", context)).ToList();
            }
            else
            {
                arg_ctx = Enumerable.Range(0, arg_shapes.Count).Select(s => context).ToList();
                aux_ctx = Enumerable.Range(0, aux_shapes.Count).Select(s => context).ToList();
            }

            //alloc space
            var arg_ndarrays = arg_types
                .Zip(arg_ctx, arg_shapes,
                    (dtype, dev, shape) =>
                        NDArray.Zeros(new Shape(shape), dev, dtype))
                .ToList();
            Dictionary<string, NDArray> grad_ndarrays = new Dictionary<string, NDArray>();
            if (grad_req != OpReqType.KNullOp)
            {
             
                for (int i = 0; i < list_arguments.Count; i++)
                {
                    var name = list_arguments[i];
                    var shape = arg_shapes[i];
                    var dev = arg_ctx[i];
                    var dtype = arg_types[i];

                    grad_ndarrays[name] = NDArray.Zeros(new Shape(shape), dev, dtype: dtype);
                }
            }

            var aux_ndarrays = aux_types
                .Zip(aux_ctx, aux_shapes,
                    (dtype, dev, shape) =>
                        NDArray.Zeros(new Shape(shape), dev, dtype))
                .ToList();

            var executor = Bind(context,
                arg_ndarrays,
                grad_ndarrays, 
                grad_req,
                aux_ndarrays,
                group_to_ctx: group2ctx);

            return executor;
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
            OpReqType grad_req,
            List<NDArray> aux_arrays,
            Dictionary<string, Context> group_to_ctx = null,
            Executor shared_exec = null)
        {
            var grad_reqs = ListArguments().ToDictionary(k => k, v => grad_req);
            return Bind(context, arg_arrays, grad_dict, grad_reqs, aux_arrays, group_to_ctx, shared_exec);
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
                        throw new Exception($"Must specify all the arguments in {arg_key}");
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
            IntPtr[] out_ptr_array = new IntPtr[out_size * 2];

            Dictionary<string, string> attr = new Dictionary<string, string>();
            for (int i = 0; i < out_size; i++)
            {
                attr.Add(Marshal.PtrToStringAnsi(out_ptr_array[i * 2]), Marshal.PtrToStringAnsi(out_ptr_array[i * 2] + 1));
            }

            return attr;

        }
        [DebuggerHidden]
        public SymbolHandle GetHandle()
        {
            return _blob_ptr.handle;
        }



    }
}
