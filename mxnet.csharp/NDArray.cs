using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Remoting.Contexts;
using System.Text;
using System.Threading.Tasks;
using mxnet.numerics.single;
using NDArrayHandle = System.IntPtr;
using FunctionHandle = System.IntPtr;

namespace mxnet.csharp
{


    public class NdBlob : IDisposable
    {
        /// <summary>
        /// construct with SymbolHandle to store
        /// </summary>
        /// <param name="handle"></param>
        public NdBlob(NDArrayHandle handle)

        {
            this.Handle = handle;
        }
        /// <summary>
        /// destructor, free the SymbolHandle
        /// </summary>
        ~NdBlob()
        {
            Dispose(false);
        }

        private void Dispose(bool disposing)
        {
            NativeMethods.MXNDArrayFree(Handle);
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
        /// the SymbolHandle to store
        /// </summary>
        public NDArrayHandle Handle { get; }
    }
    public class NdArray
    {
        private readonly bool _writable;
        private readonly NdBlob _blobPtr;
        public NdArray()
        {
            NDArrayHandle handle;
            Util.CallCheck(NativeMethods.MXNDArrayCreateNone(out handle));
            _blobPtr = new NdBlob(handle);
        }
        public NdArray(NDArrayHandle handle, bool writable = true)
        {
            _writable = writable;
            _blobPtr = new NdBlob(handle);
        }

        public NdArray(uint[] shape, Context context,
                        bool delayAlloc)
        {
            NDArrayHandle handle;
            Util.CallCheck(NativeMethods.MXNDArrayCreate(shape, (uint)shape.Length, context.DeviceType,
                           context.DeviceId, delayAlloc ? 1 : 0, out handle));
            _blobPtr = new NdBlob(handle);
        }
        public NdArray(uint[] shape)
        {
            NDArrayHandle handle;
            Util.CallCheck(NativeMethods.MXNDArrayCreate(shape, (uint)shape.Length, DeviceType.KCpu, 0, 0, out handle));
            _blobPtr = new NdBlob(handle);
        }
        public NdArray(Shape shape, Context context, bool delayAlloc)
        {
            NDArrayHandle handle;
            Util.CallCheck(NativeMethods.MXNDArrayCreate(shape.Data().ToArray(), shape.Ndim(), context.DeviceType,
                               context.DeviceId, delayAlloc ? 1 : 0, out handle));
            _blobPtr = new NdBlob(handle);
        }

        public NdArray(Shape shape, Context context, bool delayAlloc, Type dtype)
        {
            NDArrayHandle handle;
            Util.CallCheck(NativeMethods.MXNDArrayCreateEx(shape.Data().ToArray(), shape.Ndim(), context.DeviceType,
                               context.DeviceId, delayAlloc ? 1 : 0, Util.DtypeNpToMx[dtype], out handle));
            _blobPtr = new NdBlob(handle);
        }

        public NdArray(float[] data)
        {
            NDArrayHandle handle;
            Util.CallCheck(NativeMethods.MXNDArrayCreateNone(out handle));
            NativeMethods.MXNDArraySyncCopyFromCPU(handle, data, (uint)data.Length);
            _blobPtr = new NdBlob(handle);
        }
        public NdArray(float[] data, Shape shape,
                         Context context = null)

        {
            if (context == null)
            {
                context = Context.DefaultCtx;
            }

            NDArrayHandle handle;
            Util.CallCheck(NativeMethods.MXNDArrayCreate(shape.Data().ToArray(), shape.Ndim(), context.DeviceType,
                           context.DeviceId, 0, out handle));
            NativeMethods.MXNDArraySyncCopyFromCPU(handle, data, shape.Size());
            _blobPtr = new NdBlob(handle);
        }


        public void SyncCopyFromCpu(float[] data)
        {
            NativeMethods.MXNDArraySyncCopyFromCPU(_blobPtr.Handle, data, (uint)data.Length);
        }

        public float[] SyncCopyToCpu(uint size)
        {
            size = size > 0 ? size : this.Size();
            var data = new float[size];
            var datagch = GCHandle.Alloc(data, GCHandleType.Pinned);
            NativeMethods.MXNDArraySyncCopyToCPU(_blobPtr.Handle, datagch.AddrOfPinnedObject(), size);
            datagch.Free();
            return data;
        }

        public void WaitToRead()
        {
            Util.CallCheck(NativeMethods.MXNDArrayWaitToRead(_blobPtr.Handle));
        }
        public void WaitToWrite()
        {
            Util.CallCheck(NativeMethods.MXNDArrayWaitToWrite(_blobPtr.Handle));
        }
        public static void WaitAll() { Util.CallCheck(NativeMethods.MXNDArrayWaitAll()); }

        public NdArray CopyTo(NdArray other)
        {
            FunctionHandle funcHandle;
            NativeMethods.MXGetFunction("_copyto", out funcHandle);

            var input = _blobPtr.Handle;
            var output = other._blobPtr.Handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(funcHandle, ref input, new float[0], ref output));
            return other;
        }

        public NdArray CopyTo(Context ctx)
        {
            FunctionHandle funcHandle;
            NativeMethods.MXGetFunction("_copyto", out funcHandle);

            var other = new NdArray(this.GetShape(), ctx, true);
            var input = _blobPtr.Handle;
            var output = other._blobPtr.Handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(funcHandle, ref input, new float[0], ref output));
            return other;
        }

        public NdArray Slice(uint begin, uint end)
        {
            NDArrayHandle handle;
            Util.CallCheck(NativeMethods.MXNDArraySlice(Handle, begin, end, out handle));
            return new NdArray(handle);
        }

        public NdArray Reshape(Shape newShape)
        {
            NDArrayHandle handle;
            var dims = newShape.Data().Select(s => (int)s);
            Util.CallCheck(NativeMethods.MXNDArrayReshape(Handle, (int)newShape.Ndim(), dims.ToArray(), out handle));
            return new NdArray(handle);
        }

        public NdArray SetValue(float value)
        {
            FunctionHandle funcHandle;
            NativeMethods.MXGetFunction("_set_value", out funcHandle);
            float[] scalar = { value };
            IntPtr zero = IntPtr.Zero;
            var handle = _blobPtr.Handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(funcHandle, ref zero, scalar, ref handle));
            return this;
        }

        public static void SampleGaussian(float mu, float sigma, NdArray outArray)
        {
            FunctionHandle funcHandle;
            NativeMethods.MXGetFunction("_random_gaussian", out funcHandle);
            float[] scalar = { mu, sigma };
            IntPtr zero = IntPtr.Zero;
            var handle = outArray._blobPtr.Handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(funcHandle, ref zero, scalar, ref handle));
        }

        public uint Size()
        {
            return GetShape().Size();
        }

        public Shape GetShape()
        {
            IntPtr outPdata;
            uint outDim;
            NativeMethods.MXNDArrayGetShape(_blobPtr.Handle, out outDim, out outPdata);
            int[] ret = new int[outDim];
            Marshal.Copy(outPdata, ret, 0, (int)outDim);
            return new Shape(ret.Select(s => (uint)s).ToArray());
        }

        public Type GetDtype()
        {
            int outDtype;
            Util.CallCheck(NativeMethods.MXNDArrayGetDType(_blobPtr.Handle, out outDtype));
            return Util.DtypeMxToNp[outDtype];
        }


        public NdArray ArgmaxChannel()
        {
            NdArray ret = new NdArray();
            FunctionHandle funcHandle;
            NativeMethods.MXGetFunction("argmax_channel", out funcHandle);
            var input = _blobPtr.Handle;
            var output = ret._blobPtr.Handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(funcHandle, ref input, new float[0], ref output));

            return ret;
        }


        [DebuggerHidden]
        public NDArrayHandle Handle => _blobPtr.Handle;

        public static NdArray Zeros(Shape shape, Context ctx = null, Type dtype = null)
        {
            if (ctx == null)
            {
                ctx = Context.DefaultCtx;
            }
            if (dtype == null)
            {
                dtype = typeof(float);
            }

            var array = new NdArray(shape, ctx, false, dtype);
            array.SetValue(0);
            return array;
        }

        #region Numerics
        public SingleNArray AsNumerics()
        {
            var shape = GetShape();
            SingleNArray data = new SingleNArray(new numerics.nbase.Shape(shape.Data()));
            var datagch = data.GetDataGcHandle();
            IntPtr pointer = datagch.AddrOfPinnedObject();
            var s = shape.Size();
            NativeMethods.MXNDArraySyncCopyToCPU(_blobPtr.Handle, pointer, s);
            datagch.Free();
            return data;
        }
        #endregion


        public static void Save(string filename, Dictionary<string, NdArray> data)
        {
            var handles = new List<NDArrayHandle>();
            var keys = new List<string>();
            foreach (var kv in data)
            {
                handles.Add(kv.Value.Handle);

                keys.Add(kv.Key);

            }
            NativeMethods.MXNDArraySave(filename,(uint) keys.Count,handles.ToArray(),keys.ToArray());
        }

        public static void Load(string filename, out Dictionary<string, NdArray> data)
        {
            data = new Dictionary<string, NdArray>();
            uint outSize;
            IntPtr outArrPtr;
            uint outNameSize;
            IntPtr outNamesPtr;

            NativeMethods.MXNDArrayLoad(filename, out outSize, out outArrPtr, out outNameSize,out outNamesPtr);
            NDArrayHandle[] outArr = new NDArrayHandle[outSize];
            Marshal.Copy(outArrPtr, outArr, 0, (int) outSize);


            if (outNameSize == 0)
            {
                for (int i = 0; i < outArr.Length; i++)
                {
                    data.Add(i.ToString(), new NdArray(outArr[i]));
                }

            }
            else
            {
                Util.Assert(outNameSize == outSize);
                IntPtr[] outNames = new IntPtr[outNameSize];
                Marshal.Copy(outNamesPtr, outNames, 0, (int)outNameSize);

                for (int i = 0; i < outArr.Length; i++)
                {
                    var key = Marshal.PtrToStringAnsi(outNames[i]);
                    if (!string.IsNullOrEmpty(key))
                    {
                        data.Add(key, new NdArray(outArr[i]));
                    }
                }
            }
        }

        #region operator
        public static NdArray operator +(NdArray lhs, NdArray rhs)
        {
         
            //FunctionHandle funcHandle;
            //NativeMethods.MXGetFunction("_plus", out funcHandle);

            //var ret = new NdArray();
            //var input = _blobPtr.Handle;

            //var output = ret._blobPtr.Handle;
            //Util.CallCheck(NativeMethods.MXFuncInvoke(funcHandle, ref input, new float[0], ref output));
            //return other;


            return ret;
        }

        public static NdArray operator -(NdArray lhs, NdArray rhs)
        {
            return Operator._Minus(lhs, rhs);
        }

        public static NdArray operator *(NdArray lhs, NdArray rhs)
        {
            return Operator._Mul(lhs, rhs);
        }

        public static NdArray operator /(NdArray lhs, NdArray rhs)
        {
            return Operator._Div(lhs, rhs);
        }

        public static NdArray operator +(NdArray lhs, float scalar)
        {
            return Operator._PlusScalar(lhs, scalar);
        }

        public static NdArray operator +(float scalar, NdArray rhs)
        {
            return Operator._PlusScalar(rhs, scalar);
        }

        public static NdArray operator -(NdArray lhs, float scalar)
        {
            return Operator._MinusScalar(lhs, scalar);
        }

        public static NdArray operator -(float scalar, NdArray rhs)
        {
            return Operator._RMinusScalar(scalar, rhs);
        }

        public static NdArray operator *(NdArray lhs, float scalar)
        {
            return Operator._MulScalar(lhs, scalar);
        }

        public static NdArray operator *(float scalar, NdArray rhs)
        {
            return Operator._MulScalar(rhs, scalar);
        }

        public static NdArray operator /(NdArray lhs, float scalar)
        {
            return Operator._DivScalar(lhs, scalar);
        }

        public static NdArray operator /(float scalar, NdArray rhs)
        {
            return Operator._RDivScalar(scalar, rhs);
        }
        #endregion


    }

    public static class NdArrayExtension
    {
        public static NdArray Sum<TSource>(this IEnumerable<TSource> source)
        {
            return null;
        }

        public static NdArray Sum<TSource>(this IEnumerable<TSource> source, Func<TSource, NdArray> selector)
        {
            return null;
        }

    }
}
