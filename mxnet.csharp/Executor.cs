using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using NDArrayHandle = System.IntPtr;
using ExecutorHandle = System.IntPtr;

namespace mxnet.csharp
{
    public enum OpReqType
    {
        /// <summary>
        /// no operation, do not write anything
        /// </summary>
        KNullOp,
        /// <summary>
        /// write gradient to provided space
        /// </summary>
        KWriteTo,
        /// <summary>
        /// perform an inplace write,
        /// Target shares memory with one of input arguments.
        /// This option only happen when
        /// </summary>
        KWriteInplace,
        /// <summary>
        /// add to the provided space
        /// </summary>
        KAddTo
    };
    public class Executor
    {
        ExecutorHandle handle_;
        List<NDArray> outputs;
        NDArray[] arg_arrays;
        NDArray[] grad_arrays;
        NDArray[] aux_arrays;
        Symbol symbol_;

        public Executor(Symbol symbol, Context context,
                      NDArray[] argArrays,
                     NDArray[] gradArrays,
                      OpReqType[] gradReqs,
                      NDArray[] auxArrays,
                     Dictionary<string, Context> groupToCtx,
                     Executor sharedExec)
        {
            this.arg_arrays = argArrays;
            this.grad_arrays = gradArrays;
            this.aux_arrays = auxArrays;
            this.symbol_ = symbol;

            List<NDArrayHandle> argHandles = new List<NDArrayHandle>();
            List<NDArrayHandle> gradHandles = new List<NDArrayHandle>();
            List<NDArrayHandle> auxHandles = new List<NDArrayHandle>();

            foreach (var array in argArrays)
            {
                argHandles.Add(array.GetHandle());
            }
            foreach (var array in gradArrays)
            {
                gradHandles.Add(array.GetHandle());
            }
            foreach (var array in auxArrays)
            {
                auxHandles.Add(array.GetHandle());
            }

            List<uint> gradReqsUint = new List<uint>();
            foreach (var s in gradReqs) { gradReqsUint.Add((uint)s); }

            List<string> mapKeys = new List<string>();
            List<int> devTypes = new List<int>();
            List<int> devIds = new List<int>();
            foreach (var s in groupToCtx)
            {
                mapKeys.Add(s.Key);
                devTypes.Add((int)s.Value.GetDeviceType());
                devIds.Add(s.Value.GetDeviceId());
            }

            ExecutorHandle sharedExecHandle =
                sharedExec?.handle_ ?? IntPtr.Zero;

            Debug.Assert(NativeMethods.MXExecutorBindEX(
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
                out handle_) == 0);

            uint outSize;
            NDArrayHandle outArrayPtr;
            Debug.Assert(NativeMethods.MXExecutorOutputs(handle_, out outSize, out outArrayPtr) == 0);
            IntPtr[] outArray = new IntPtr[outSize];
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
            IntPtr outputPtr;
            NativeMethods.MXExecutorPrint(handle_, out outputPtr);
            return Marshal.PtrToStringAnsi(outputPtr);
        }



        public void UpdateAll(Optimizer opt, float lr, float wd,
                                 int argUpdateBegin = 1, int argUpdateEnd = -1)
        {
            argUpdateEnd = argUpdateEnd < 0 ? arg_arrays.Length - 1 : argUpdateEnd;
            for (int i = argUpdateBegin; i < argUpdateEnd; ++i)
            {
                opt.Update(i, arg_arrays[i], grad_arrays[i], lr, wd);
            }
        }



        /// <summary>
        /// Perform a Forward operation of Operator
        /// After this operation, user can get the result by using function head.
        /// </summary>
        /// <param name="is_train"></param>
        public void Forward(bool is_train)
        {
            NativeMethods.MXExecutorForward(handle_, is_train ? 1 : 0);
            uint out_size;
            IntPtr out_array_ptr;
            Debug.Assert(NativeMethods.MXExecutorOutputs(handle_, out out_size, out out_array_ptr) == 0);
            IntPtr[] out_array = new IntPtr[out_size];

            Marshal.Copy(out_array_ptr, out_array, 0, (int)out_size);
            for (int i = 0; i < out_size; ++i)
            {
                outputs[i] = new NDArray(out_array[i]);
            }
        }

        /// <summary>
        /// Perform a Backward operation of the Operator.
        /// This must be called after Forward.
        /// After this operation, NDArrays specified by grad_in_args_store will be
        /// updated accordingly.
        /// User is allowed to pass in an empty Array if the head node is
        /// loss function and head gradeitn is not needed.
        /// </summary>
        /// <param name="headGrads">the gradient of head nodes to be backproped.</param>
        public void Backward(List<NDArray> headGrads = null)
        {

            if (headGrads == null)
            {
                headGrads = new List<NDArray>();
            }
            List<NDArray> newHeadGrads = new List<NDArray>();
            foreach (var d in headGrads)
            {
                newHeadGrads.Add(new NDArray(d.GetHandle()));
            }
            if (newHeadGrads.Count > 0)
            {
                IntPtr[] ptrs = newHeadGrads.Select(s => s.GetHandle()).ToArray();

                NativeMethods.MXExecutorBackward(handle_, (uint)newHeadGrads.Count, ptrs);
            }
            else
            {
                NativeMethods.MXExecutorBackward(handle_, 0, IntPtr.Zero);
            }
        }
    }
}
