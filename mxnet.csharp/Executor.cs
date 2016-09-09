using System;
using System.Collections;
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
        private readonly ExecutorHandle _handle;
        public IList<NdArray> Outputs { get; } = new List<NdArray>();
        public IList<NdArray> ArgArrays { get; }
        public IList<NdArray> GradArrays { get; }
        public IList<NdArray> AuxArrays { get; }

        public Dictionary<string, NdArray> ArgDict { get; private set; }
        public Dictionary<string, NdArray> GradDict { get; private set; }

        private Symbol _symbol;
        private readonly Dictionary<string, NdArray> _auxDict;

        public Executor(Symbol symbol, Context context,
            IList<NdArray> argArrays,
            IList<NdArray> gradArrays,
            IList<OpReqType> gradReqs,
            IList<NdArray> auxArrays,
            Dictionary<string, Context> groupToCtx = null,
            Executor sharedExec = null)
        {
            if (groupToCtx == null)
            {
                groupToCtx = new Dictionary<string, Context>();
            }
            this.ArgArrays = argArrays;
            this.GradArrays = gradArrays;
            this.AuxArrays = auxArrays;
            this._symbol = symbol;

            var argName = symbol.ListArguments();

            ArgDict = argName.Zip(argArrays, (name, arg) => new { name, arg })
                .ToDictionary(k => k.name, v => v.arg);
            GradDict = argName.Zip(gradArrays, (name, arg) => new { name, arg })
                .ToDictionary(k => k.name, v => v.arg);

            _auxDict = symbol.ListAuxiliaryStates().Zip(auxArrays, (name, arg) => new { name, arg })
                 .ToDictionary(k => k.name, v => v.arg);

            var argHandles = new List<NDArrayHandle>();
            var gradHandles = new List<NDArrayHandle>();
            var auxHandles = new List<NDArrayHandle>();

            foreach (var array in argArrays)
            {
                argHandles.Add(array.Handle);
            }
            foreach (var array in gradArrays)
            {
                if (array == null)
                {
                    gradHandles.Add(IntPtr.Zero);
                }
                else
                {
                    gradHandles.Add(array.Handle);
                }
            }
            foreach (var array in auxArrays)
            {
                auxHandles.Add(array.Handle);
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
                devTypes.Add((int)s.Value.DeviceType);
                devIds.Add(s.Value.DeviceId);
            }

            var sharedExecHandle =
                sharedExec?._handle ?? NDArrayHandle.Zero;

            Util.CallCheck(NativeMethods.MXExecutorBindEX(
                symbol.get_handle(),
                (int)context.DeviceType,
                context.DeviceId,
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
                out _handle) );

            uint outSize;
            NDArrayHandle outArrayPtr;
            Util.CallCheck(NativeMethods.MXExecutorOutputs(_handle, out outSize, out outArrayPtr) );
            var outArray = new NDArrayHandle[outSize];
            if (outSize > 0)
            {
                Marshal.Copy(outArrayPtr, outArray, 0, (int)outSize);
            }
            for (uint i = 0; i < outSize; ++i)
            {
                Outputs.Add(new NdArray(outArray[i]));
            }
        }

        public string DebugStr()
        {
            NDArrayHandle outputPtr;
            NativeMethods.MXExecutorPrint(_handle, out outputPtr);
            return Marshal.PtrToStringAnsi(outputPtr);
        }



        /// <summary>
        ///     Perform a Forward operation of Operator
        ///     After this operation, user can get the result by using function head.
        /// </summary>
        /// <param name="isTrain"></param>
        public void Forward(bool isTrain)
        {
            NativeMethods.MXExecutorForward(_handle, isTrain ? 1 : 0);
            uint outSize;
            NDArrayHandle outArrayPtr;
            Util.CallCheck(NativeMethods.MXExecutorOutputs(_handle, out outSize, out outArrayPtr) );
            var outArray = new NDArrayHandle[outSize];

            Marshal.Copy(outArrayPtr, outArray, 0, (int)outSize);
            for (var i = 0; i < outSize; ++i)
            {
                Outputs[i] = new NdArray(outArray[i]);
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
        public void Backward(IList<NdArray> headGrads = null)
        {
            if (headGrads == null)
            {
                headGrads = new List<NdArray>();
            }
            var newHeadGrads = new List<NdArray>();
            foreach (var d in headGrads)
            {
                newHeadGrads.Add(new NdArray(d.Handle));
            }
            if (newHeadGrads.Count > 0)
            {
                var ptrs = newHeadGrads.Select(s => s.Handle).ToArray();

                NativeMethods.MXExecutorBackward(_handle, (uint)newHeadGrads.Count, ptrs);
            }
            else
            {
                NativeMethods.MXExecutorBackward(_handle, 0, NDArrayHandle.Zero);
            }
        }

        public void SetMonitorCallback(ExecutorMonitorCallback callback)
        {
            NativeMethods.MXExecutorSetMonitorCallback(_handle, callback, IntPtr.Zero);
        }
        public void CopyParamsFrom(Dictionary<string, NdArray> argParams,
            Dictionary<string, NdArray> auxParams = null, bool allowExtraParams = false)
        {
            foreach (var kv in argParams)
            {
                if (ArgDict.ContainsKey(kv.Key))
                {
                    kv.Value.CopyTo(ArgDict[kv.Key]);
                }
                else
                {
                    if (!allowExtraParams)
                    {
                        throw new Exception($"Find name \"{kv.Key}\" that is not in the arguments");
                    }

                }

            }
            if (auxParams != null)
            {
                foreach (var kv in auxParams)
                {
                    if (_auxDict.ContainsKey(kv.Key))
                    {
                        kv.Value.CopyTo(_auxDict[kv.Key]);
                    }
                    else
                    {
                        if (!allowExtraParams)
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
            NativeMethods.MXExecutorFree(_handle);
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
