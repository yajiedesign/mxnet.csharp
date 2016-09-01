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
        public List<NDArray> outputs { get; } = new List<NDArray>();
        public List<NDArray> arg_arrays { get; }
        public List<NDArray> grad_arrays { get; }
        public List<NDArray> aux_arrays { get; }

        public Dictionary<string, NDArray> arg_dict { get; private set; }
        public Dictionary<string, NDArray> grad_dict { get; private set; }

        private Symbol symbol_;
        private Dictionary<string, NDArray> aux_dict;

        public Executor(Symbol symbol, Context context,
            List<NDArray> argArrays,
            List<NDArray> gradArrays,
            List<OpReqType> gradReqs,
            List<NDArray> auxArrays,
            Dictionary<string, Context> groupToCtx = null,
            Executor sharedExec = null)
        {
            if (groupToCtx == null)
            {
                groupToCtx = new Dictionary<string, Context>();
            }
            this.arg_arrays = argArrays;
            this.grad_arrays = gradArrays;
            this.aux_arrays = auxArrays;
            this.symbol_ = symbol;

            var arg_name = symbol.ListArguments();

            arg_dict = arg_name.Zip(argArrays, (name, arg) => new { name, arg })
                .ToDictionary(k => k.name, v => v.arg);
            grad_dict = arg_name.Zip(gradArrays, (name, arg) => new { name, arg })
                .ToDictionary(k => k.name, v => v.arg);

            aux_dict = symbol.ListAuxiliaryStates().Zip(auxArrays, (name, arg) => new { name, arg })
                 .ToDictionary(k => k.name, v => v.arg);

            var argHandles = new List<NDArrayHandle>();
            var gradHandles = new List<NDArrayHandle>();
            var auxHandles = new List<NDArrayHandle>();

            foreach (var array in argArrays)
            {
                argHandles.Add(array.GetHandle());
            }
            foreach (var array in gradArrays)
            {
                if (array == null)
                {
                    gradHandles.Add(IntPtr.Zero);
                }
                else
                {
                    gradHandles.Add(array.GetHandle());
                }
            }
            foreach (var array in auxArrays)
            {
                auxHandles.Add(array.GetHandle());
            }

            var gradReqsUint = new List<uint>();
            foreach (var s in gradReqs)
            {
                gradReqsUint.Add((uint)s);
            }

            var mapKeys = new List<string>();
            var devTypes = new List<int>();
            var devIds = new List<int>();
            foreach (var s in groupToCtx)
            {
                mapKeys.Add(s.Key);
                devTypes.Add((int)s.Value.GetDeviceType());
                devIds.Add(s.Value.GetDeviceId());
            }

            var sharedExecHandle =
                sharedExec?.handle_ ?? NDArrayHandle.Zero;

            Util.call_check(NativeMethods.MXExecutorBindEX(
                symbol.GetHandle(),
                (int)context.GetDeviceType(),
                context.GetDeviceId(),
                (uint)groupToCtx.Count,
                mapKeys.ToArray(),
                devTypes.ToArray(),
                devIds.ToArray(),
                (uint)argHandles.Count,
                argHandles.ToArray(),
                gradHandles.ToArray(),
                gradReqsUint.ToArray(),
                (uint)auxHandles.Count,
                auxHandles.ToArray(),
                sharedExecHandle,
                out handle_) );

            uint outSize;
            NDArrayHandle outArrayPtr;
            Util.call_check(NativeMethods.MXExecutorOutputs(handle_, out outSize, out outArrayPtr) );
            var outArray = new NDArrayHandle[outSize];
            if (outSize > 0)
            {
                Marshal.Copy(outArrayPtr, outArray, 0, (int)outSize);
            }
            for (uint i = 0; i < outSize; ++i)
            {
                outputs.Add(new NDArray(outArray[i]));
            }
        }

        public string DebugStr()
        {
            NDArrayHandle outputPtr;
            NativeMethods.MXExecutorPrint(handle_, out outputPtr);
            return Marshal.PtrToStringAnsi(outputPtr);
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
                outputs[i] = new NDArray(out_array[i]);
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
            var newHeadGrads = new List<NDArray>();
            foreach (var d in headGrads)
            {
                newHeadGrads.Add(new NDArray(d.GetHandle()));
            }
            if (newHeadGrads.Count > 0)
            {
                var ptrs = newHeadGrads.Select(s => s.GetHandle()).ToArray();

                NativeMethods.MXExecutorBackward(handle_, (uint)newHeadGrads.Count, ptrs);
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
                if (arg_dict.ContainsKey(kv.Key))
                {
                    kv.Value.CopyTo(arg_dict[kv.Key]);
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
                    if (aux_dict.ContainsKey(kv.Key))
                    {
                        kv.Value.CopyTo(aux_dict[kv.Key]);
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
