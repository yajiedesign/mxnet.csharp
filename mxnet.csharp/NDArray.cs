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


    public class NDBlob : IDisposable
    {
        /// <summary>
        /// construct with SymbolHandle to store
        /// </summary>
        /// <param name="handle"></param>
        public NDBlob(NDArrayHandle handle)

        {
            this.handle = handle;
        }
        /// <summary>
        /// destructor, free the SymbolHandle
        /// </summary>
        ~NDBlob()
        {
            Dispose(false);
        }

        private void Dispose(bool disposing)
        {
            NativeMethods.MXNDArrayFree(handle);
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
        public NDArrayHandle handle { get; }
    }
    public class NDArray
    {
        private readonly bool _writable;
        private readonly NDBlob _blob_ptr;
        public NDArray()
        {
            NDArrayHandle handle;
            Util.CallCheck(NativeMethods.MXNDArrayCreateNone(out handle));
            _blob_ptr = new NDBlob(handle);
        }
        public NDArray(NDArrayHandle handle, bool writable = true)
        {
            _writable = writable;
            _blob_ptr = new NDBlob(handle);
        }

        public NDArray(uint[] shape, Context context,
                        bool delay_alloc)
        {
            NDArrayHandle handle;
            Util.CallCheck(NativeMethods.MXNDArrayCreate(shape, (uint)shape.Length, context.device_type,
                           context.device_id, delay_alloc ? 1 : 0, out handle));
            _blob_ptr = new NDBlob(handle);
        }
        public NDArray(uint[] shape)
        {
            NDArrayHandle handle;
            Util.CallCheck(NativeMethods.MXNDArrayCreate(shape, (uint)shape.Length, DeviceType.KCpu, 0, 0, out handle));
            _blob_ptr = new NDBlob(handle);
        }
        public NDArray(Shape shape, Context context, bool delay_alloc)
        {
            NDArrayHandle handle;
            Util.CallCheck(NativeMethods.MXNDArrayCreate(shape.Data().ToArray(), shape.Ndim(), context.device_type,
                               context.device_id, delay_alloc ? 1 : 0, out handle));
            _blob_ptr = new NDBlob(handle);
        }

        public NDArray(Shape shape, Context context, bool delay_alloc, Type dtype)
        {
            NDArrayHandle handle;
            Util.CallCheck(NativeMethods.MXNDArrayCreateEx(shape.Data().ToArray(), shape.Ndim(), context.device_type,
                               context.device_id, delay_alloc ? 1 : 0, Util.DtypeNpToMX[dtype], out handle));
            _blob_ptr = new NDBlob(handle);
        }

        public NDArray(float[] data)
        {
            NDArrayHandle handle;
            Util.CallCheck(NativeMethods.MXNDArrayCreateNone(out handle));
            NativeMethods.MXNDArraySyncCopyFromCPU(handle, data, (uint)data.Length);
            _blob_ptr = new NDBlob(handle);
        }
        public NDArray(float[] data, Shape shape,
                         Context context = null)

        {
            if (context == null)
            {
                context = Context.default_ctx;
            }

            NDArrayHandle handle;
            Util.CallCheck(NativeMethods.MXNDArrayCreate(shape.Data().ToArray(), shape.Ndim(), context.device_type,
                           context.device_id, 0, out handle));
            NativeMethods.MXNDArraySyncCopyFromCPU(handle, data, shape.Size());
            _blob_ptr = new NDBlob(handle);
        }


        public void sync_copy_from_cpu(float[] data)
        {
            NativeMethods.MXNDArraySyncCopyFromCPU(_blob_ptr.handle, data, (uint)data.Length);
        }

        public float[] sync_copy_to_cpu(uint size)
        {
            size = size > 0 ? size : this.Size();
            var data = new float[size];
            var datagch = GCHandle.Alloc(data, GCHandleType.Pinned);
            NativeMethods.MXNDArraySyncCopyToCPU(_blob_ptr.handle, datagch.AddrOfPinnedObject(), size);
            datagch.Free();
            return data;
        }

        public void wait_to_read()
        {
            Util.CallCheck(NativeMethods.MXNDArrayWaitToRead(_blob_ptr.handle));
        }
        public void wait_to_write()
        {
            Util.CallCheck(NativeMethods.MXNDArrayWaitToWrite(_blob_ptr.handle));
        }
        public static void wait_all() { Util.CallCheck(NativeMethods.MXNDArrayWaitAll()); }

        public NDArray copy_to(NDArray other)
        {
            FunctionHandle func_handle;
            NativeMethods.MXGetFunction("_copyto", out func_handle);

            var input = _blob_ptr.handle;
            var output = other._blob_ptr.handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(func_handle, ref input, new float[0], ref output));
            return other;
        }

        public NDArray Slice(uint begin, uint end)
        {
            NDArrayHandle handle;
            Util.CallCheck(NativeMethods.MXNDArraySlice(get_handle(), begin, end, out handle));
            return new NDArray(handle);
        }

        public NDArray Reshape(Shape new_shape)
        {
            NDArrayHandle handle;
            var dims = new_shape.Data().Select(s => (int)s);
            Util.CallCheck(NativeMethods.MXNDArrayReshape(get_handle(), (int)new_shape.Ndim(), dims.ToArray(), out handle));
            return new NDArray(handle);
        }

        public NDArray set_value(float value)
        {
            FunctionHandle func_handle;
            NativeMethods.MXGetFunction("_set_value", out func_handle);
            float[] scalar = { value };
            IntPtr zero = IntPtr.Zero;
            var handle = _blob_ptr.handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(func_handle, ref zero, scalar, ref handle));
            return this;
        }

        public static void sample_gaussian(float mu, float sigma, NDArray out_array)
        {
            FunctionHandle func_handle;
            NativeMethods.MXGetFunction("_random_gaussian", out func_handle);
            float[] scalar = { mu, sigma };
            IntPtr zero = IntPtr.Zero;
            var handle = out_array._blob_ptr.handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(func_handle, ref zero, scalar, ref handle));
        }

        public uint Size()
        {
            return get_shape().Size();
        }

        public Shape get_shape()
        {
            IntPtr out_pdata;
            uint out_dim;
            NativeMethods.MXNDArrayGetShape(_blob_ptr.handle, out out_dim, out out_pdata);
            int[] ret = new int[out_dim];
            Marshal.Copy(out_pdata, ret, 0, (int)out_dim);
            return new Shape(ret.Select(s => (uint)s).ToArray());
        }

        public Type get_dtype()
        {
            int out_dtype;
            Util.CallCheck(NativeMethods.MXNDArrayGetDType(_blob_ptr.handle, out out_dtype));
            return Util.DtypeMXToNp[out_dtype];
        }


        public NDArray argmax_channel()
        {
            NDArray ret = new NDArray();
            FunctionHandle func_handle;
            NativeMethods.MXGetFunction("argmax_channel", out func_handle);
            var input = _blob_ptr.handle;
            var output = ret._blob_ptr.handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(func_handle, ref input, new float[0], ref output));

            return ret;
        }


        [DebuggerHidden]
        public NDArrayHandle get_handle() { return _blob_ptr.handle; }

        public static NDArray Zeros(Shape shape, Context ctx = null, Type dtype = null)
        {
            if (ctx == null)
            {
                ctx = Context.default_ctx;
            }
            if (dtype == null)
            {
                dtype = typeof(float);
            }

            var array = new NDArray(shape, ctx, false, dtype);
            array.set_value(0);
            return array;
        }

        #region Numerics
        public SingleNArray as_numerics()
        {
            var shape = get_shape();
            SingleNArray data = new SingleNArray(new numerics.nbase.Shape(shape.Data()));
            var datagch = data.GetDataGcHandle();
            IntPtr pointer = datagch.AddrOfPinnedObject();
            var s = shape.Size();
            NativeMethods.MXNDArraySyncCopyToCPU(_blob_ptr.handle, pointer, s);
            datagch.Free();
            return data;
        }
        #endregion


    }
}
