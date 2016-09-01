using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using mxnet.csharp.optimizer;
using NDArrayHandle = System.IntPtr;
using ExecutorHandle = System.IntPtr;

namespace mxnet.csharp
{
    public enum OpReqType
    {
        /// <summary>
        ///     no operation, do not write anything
        /// </summary>
        KNullOp,

        /// <summary>
        ///     write gradient to provided space
        /// </summary>
        KWriteTo,

        /// <summary>
        ///     perform an inplace write,
        ///     Target shares memory with one of input arguments.
        ///     This option only happen when
        /// </summary>
        KWriteInplace,

        /// <summary>
        ///     add to the provided space
        /// </summary>
        KAddTo
    }

    public class Executor : IDisposable
    {
        private readonly ExecutorHandle handle_;
        public List<NDArray> Outputs { get; } = new List<NDArray>();
        public List<NDArray> Arg_arrays { get; }
        public List<NDArray> Grad_arrays { get; }
        public List<NDArray> Aux_arrays { get; }

        public Dictionary<string, NDArray> Arg_dict { get; private set; }
        public Dictionary<string, NDArray> Grad_dict { get; private set; }

        private Symbol _symbol_;
        private readonly Dictionary<string, NDArray> _aux_dict;

        public Executor(Symbol symbol, Context context,
            List<NDArray> arg_arrays,
            List<NDArray> grad_arrays,
            List<OpReqType> grad_reqs,
            List<NDArray> aux_arrays,
            Dictionary<string, Context> group_to_ctx = null,
            Executor shared_exec = null)
        {
            if (group_to_ctx == null)
            {
                group_to_ctx = new Dictionary<string, Context>();
            }
            this.Arg_arrays = arg_arrays;
            this.Grad_arrays = grad_arrays;
            this.Aux_arrays = aux_arrays;
            this._symbol_ = symbol;

            var arg_name = symbol.ListArguments();

            Arg_dict = arg_name.Zip(arg_arrays, (name, arg) => new { name, arg })
                .ToDictionary(k => k.name, v => v.arg);
            Grad_dict = arg_name.Zip(grad_arrays, (name, arg) => new { name, arg })
                .ToDictionary(k => k.name, v => v.arg);

            _aux_dict = symbol.ListAuxiliaryStates().Zip(aux_arrays, (name, arg) => new { name, arg })
                 .ToDictionary(k => k.name, v => v.arg);

            var arg_handles = new List<NDArrayHandle>();
            var grad_handles = new List<NDArrayHandle>();
            var aux_handles = new List<NDArrayHandle>();

            foreach (var array in arg_arrays)
            {
                arg_handles.Add(array.Get_handle());
            }
            foreach (var array in grad_arrays)
            {
                if (array == null)
                {
                    grad_handles.Add(IntPtr.Zero);
                }
                else
                {
                    grad_handles.Add(array.Get_handle());
                }
            }
            foreach (var array in aux_arrays)
            {
                aux_handles.Add(array.Get_handle());
            }

            var grad_reqs_uint = new List<uint>();
            foreach (var s in grad_reqs)
            {
                grad_reqs_uint.Add((uint)s);
            }

            var map_keys = new List<string>();
            var dev_types = new List<int>();
            var dev_ids = new List<int>();
            foreach (var s in group_to_ctx)
            {
                map_keys.Add(s.Key);
                dev_types.Add((int)s.Value.Get_device_type());
                dev_ids.Add(s.Value.Get_device_id());
            }

            var shared_exec_handle =
                shared_exec?.handle_ ?? NDArrayHandle.Zero;

            Util.call_check(NativeMethods.MXExecutorBindEX(
                symbol.GetHandle(),
                (int)context.Get_device_type(),
                context.Get_device_id(),
                (uint)group_to_ctx.Count,
                map_keys.ToArray(),
                dev_types.ToArray(),
                dev_ids.ToArray(),
                (uint)arg_handles.Count,
                arg_handles.ToArray(),
                grad_handles.ToArray(),
                grad_reqs_uint.ToArray(),
                (uint)aux_handles.Count,
                aux_handles.ToArray(),
                shared_exec_handle,
                out handle_) );

            uint out_size;
            NDArrayHandle out_array_ptr;
            Util.call_check(NativeMethods.MXExecutorOutputs(handle_, out out_size, out out_array_ptr) );
            var out_array = new NDArrayHandle[out_size];
            if (out_size > 0)
            {
                Marshal.Copy(out_array_ptr, out_array, 0, (int)out_size);
            }
            for (uint i = 0; i < out_size; ++i)
            {
                Outputs.Add(new NDArray(out_array[i]));
            }
        }

        public string Debug_str()
        {
            NDArrayHandle output_ptr;
            NativeMethods.MXExecutorPrint(handle_, out output_ptr);
            return Marshal.PtrToStringAnsi(output_ptr);
        }



        /// <summary>
        ///     Perform a Forward operation of Operator
        ///     After this operation, user can get the result by using function head.
        /// </summary>
        /// <param name="is_train"></param>
        public void Forward(bool is_train)
        {
            NativeMethods.MXExecutorForward(handle_, is_train ? 1 : 0);
            uint out_size;
            NDArrayHandle out_array_ptr;
            Util.call_check(NativeMethods.MXExecutorOutputs(handle_, out out_size, out out_array_ptr) );
            var out_array = new NDArrayHandle[out_size];

            Marshal.Copy(out_array_ptr, out_array, 0, (int)out_size);
            for (var i = 0; i < out_size; ++i)
            {
                Outputs[i] = new NDArray(out_array[i]);
            }
        }

        /// <summary>
        ///     Perform a Backward operation of the Operator.
        ///     This must be called after Forward.
        ///     After this operation, NDArrays specified by grad_in_args_store will be
        ///     updated accordingly.
        ///     User is allowed to pass in an empty Array if the head node is
        ///     loss function and head gradeitn is not needed.
        /// </summary>
        /// <param name="headGrads">the gradient of head nodes to be backproped.</param>
        public void Backward(List<NDArray> headGrads = null)
        {
            if (headGrads == null)
            {
                headGrads = new List<NDArray>();
            }
            var new_head_grads = new List<NDArray>();
            foreach (var d in headGrads)
            {
                new_head_grads.Add(new NDArray(d.Get_handle()));
            }
            if (new_head_grads.Count > 0)
            {
                var ptrs = new_head_grads.Select(s => s.Get_handle()).ToArray();

                NativeMethods.MXExecutorBackward(handle_, (uint)new_head_grads.Count, ptrs);
            }
            else
            {
                NativeMethods.MXExecutorBackward(handle_, 0, NDArrayHandle.Zero);
            }
        }

        public void set_monitor_callback(ExecutorMonitorCallback callback)
        {
            NativeMethods.MXExecutorSetMonitorCallback(handle_, callback, IntPtr.Zero);
        }
        public void copy_params_from(Dictionary<string, NDArray> argParams, Dictionary<string, NDArray> auxParams = null, bool allow_extra_params = false)
        {
            foreach (var kv in argParams)
            {
                if (Arg_dict.ContainsKey(kv.Key))
                {
                    kv.Value.Copy_to(Arg_dict[kv.Key]);
                }
                else
                {
                    if (!allow_extra_params)
                    {
                        throw new Exception($"Find name \"{kv.Key}\" that is not in the arguments");
                    }

                }

            }
            if (auxParams != null)
            {
                foreach (var kv in auxParams)
                {
                    if (_aux_dict.ContainsKey(kv.Key))
                    {
                        kv.Value.Copy_to(_aux_dict[kv.Key]);
                    }
                    else
                    {
                        if (!allow_extra_params)
                        {
                            throw new Exception($"Find name \"{kv.Key}\" that is not the auxiliary states");
                        }

                    }
                }
            }

        }

        /// <summary>
        /// destructor, free the SymbolHandle
        /// </summary>
        ~Executor()
        {
            Dispose(false);
        }

        private void Dispose(bool disposing)
        {
            NativeMethods.MXExecutorFree(handle_);
            if (disposing)
            {
                GC.SuppressFinalize(this);
            }
        }

        public void Dispose()
        {
            Dispose(true);
        }



    }
}
