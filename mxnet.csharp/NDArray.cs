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
    public enum DeviceType
    {
        KCpu = 1,
        KGpu = 2,
        KCpuPinned = 3
    };


    public class Context
    {
        private readonly DeviceType _type;
        private readonly int _id;


        /// <summary>
        /// Context constructor
        /// </summary>
        /// <param name="type">type of the device</param>
        /// <param name="id">id of the device</param>
        public Context(DeviceType type, int id)
        {
            _type = type;
            _id = id;
        }

        public static Context Default_ctx { get; set; } = new Context(DeviceType.KCpu, 0);

        /// <summary>
        /// 
        /// </summary>
        /// <returns>the type of the device</returns>
        [DebuggerHidden]
        public DeviceType Get_device_type() { return _type; }

        /// <summary>
        /// 
        /// </summary>
        /// <returns>the id of the device</returns>
        [DebuggerHidden]
        public int Get_device_id() { return _id; }


        /// <summary>
        /// Return a GPU context
        /// </summary>
        /// <param name="device_id">id of the device</param>
        /// <returns>the corresponding GPU context</returns>
        public static Context Gpu(int device_id = 0)
        {
            return new Context(DeviceType.KGpu, device_id);
        }


        /// <summary>
        /// Return a CPU context
        /// </summary>
        /// <param name="device_id">id of the device. this is not needed by CPU</param>
        /// <returns>the corresponding CPU context</returns>
        public static Context Cpu(int device_id = 0)
        {
            return new Context(DeviceType.KCpu, device_id);
        }



    };
    public class NDBlob : IDisposable
    {
        /// <summary>
        /// construct with SymbolHandle to store
        /// </summary>
        /// <param name="handle"></param>
        public NDBlob(NDArrayHandle handle)

        {
            Handle = handle;
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
    public class NDArray
    {
        private readonly bool _writable;
        private readonly NDBlob _blob_ptr;
        public NDArray()
        {
            NDArrayHandle handle;
            Util.call_check(NativeMethods.MXNDArrayCreateNone(out handle));
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
            Util.call_check(NativeMethods.MXNDArrayCreate(shape, (uint)shape.Length, context.Get_device_type(),
                           context.Get_device_id(), delay_alloc ? 1 : 0, out handle));
            _blob_ptr = new NDBlob(handle);
        }
        public NDArray(uint[] shape)
        {
            NDArrayHandle handle;
            Util.call_check(NativeMethods.MXNDArrayCreate(shape, (uint)shape.Length, DeviceType.KCpu, 0, 0, out handle));
            _blob_ptr = new NDBlob(handle);
        }
        public NDArray(Shape shape, Context context, bool delay_alloc)
        {
            NDArrayHandle handle;
            Util.call_check(NativeMethods.MXNDArrayCreate(shape.data().ToArray(), shape.ndim(), context.Get_device_type(),
                               context.Get_device_id(), delay_alloc ? 1 : 0, out handle));
            _blob_ptr = new NDBlob(handle);
        }

        public NDArray(Shape shape, Context context, bool delay_alloc, Type dtype)
        {
            NDArrayHandle handle;
            Util.call_check(NativeMethods.MXNDArrayCreateEx(shape.data().ToArray(), shape.ndim(), context.Get_device_type(),
                               context.Get_device_id(), delay_alloc ? 1 : 0, Util._DTYPE_NP_TO_MX[dtype], out handle));
            _blob_ptr = new NDBlob(handle);
        }

        public NDArray(float[] data)
        {
            NDArrayHandle handle;
            Util.call_check(NativeMethods.MXNDArrayCreateNone(out handle));
            NativeMethods.MXNDArraySyncCopyFromCPU(handle, data, (uint)data.Length);
            _blob_ptr = new NDBlob(handle);
        }
        public NDArray(float[] data, Shape shape,
                         Context context = null)

        {
            if (context == null)
            {
                context = Context.Default_ctx;
            }

            NDArrayHandle handle;
            Util.call_check(NativeMethods.MXNDArrayCreate(shape.data().ToArray(), shape.ndim(), context.Get_device_type(),
                           context.Get_device_id(), 0, out handle));
            NativeMethods.MXNDArraySyncCopyFromCPU(handle, data, shape.Size());
            _blob_ptr = new NDBlob(handle);
        }


        public void Sync_copy_from_cpu(float[] data)
        {
            NativeMethods.MXNDArraySyncCopyFromCPU(_blob_ptr.Handle, data, (uint)data.Length);
        }

        public float[] Sync_copy_to_cpu(uint size)
        {
            size = size > 0 ? size : Size();
            var data = new float[size];
            var datagch = GCHandle.Alloc(data, GCHandleType.Pinned);
            NativeMethods.MXNDArraySyncCopyToCPU(_blob_ptr.Handle, datagch.AddrOfPinnedObject(), size);
            datagch.Free();
            return data;
        }

        public void Wait_to_read()
        {
            Util.call_check(NativeMethods.MXNDArrayWaitToRead(_blob_ptr.Handle));
        }
        public void Wait_to_write()
        {
            Util.call_check(NativeMethods.MXNDArrayWaitToWrite(_blob_ptr.Handle));
        }
        public static void Wait_all() { Util.call_check(NativeMethods.MXNDArrayWaitAll()); }

        public NDArray Copy_to(NDArray other)
        {
            FunctionHandle func_handle;
            NativeMethods.MXGetFunction("_copyto", out func_handle);

            var input = _blob_ptr.Handle;
            var output = other._blob_ptr.Handle;
            Util.call_check(NativeMethods.MXFuncInvoke(func_handle, ref input, new float[0], ref output));
            return other;
        }

        public NDArray Slice(uint begin, uint end)
        {
            NDArrayHandle handle;
            Util.call_check(NativeMethods.MXNDArraySlice(Get_handle(), begin, end, out handle));
            return new NDArray(handle);
        }

        public NDArray Reshape(Shape new_shape)
        {
            NDArrayHandle handle;
            var dims = new_shape.data().Select(s => (int)s);
            Util.call_check(NativeMethods.MXNDArrayReshape(Get_handle(), (int)new_shape.ndim(), dims.ToArray(), out handle));
            return new NDArray(handle);
        }

        public NDArray Set_value(float value)
        {
            FunctionHandle func_handle;
            NativeMethods.MXGetFunction("_set_value", out func_handle);
            float[] scalar = { value };
            IntPtr zero = IntPtr.Zero;
            var handle = _blob_ptr.Handle;
            Util.call_check(NativeMethods.MXFuncInvoke(func_handle, ref zero, scalar, ref handle));
            return this;
        }

        public static void Sample_gaussian(float mu, float sigma, NDArray out_array)
        {
            FunctionHandle func_handle;
            NativeMethods.MXGetFunction("_random_gaussian", out func_handle);
            float[] scalar = { mu, sigma };
            IntPtr Zero = IntPtr.Zero;
            var handle = out_array._blob_ptr.Handle;
            Util.call_check(NativeMethods.MXFuncInvoke(func_handle, ref Zero, scalar, ref handle));
        }

        public uint Size()
        {
            return Get_shape().Size();
        }

        public Shape Get_shape()
        {
            IntPtr out_pdata;
            uint out_dim;
            NativeMethods.MXNDArrayGetShape(_blob_ptr.Handle, out out_dim, out out_pdata);
            int[] ret = new int[out_dim];
            Marshal.Copy(out_pdata, ret, 0, (int)out_dim);
            return new Shape(ret.Select(s => (uint)s).ToArray());
        }

        public Type Get_dtype()
        {
            int out_dtype;
            Util.call_check(NativeMethods.MXNDArrayGetDType(_blob_ptr.Handle, out out_dtype));
            return Util._DTYPE_MX_TO_NP[out_dtype];
        }


        public NDArray argmax_channel()
        {
            NDArray ret = new NDArray();
            FunctionHandle func_handle;
            NativeMethods.MXGetFunction("argmax_channel", out func_handle);
            var input = _blob_ptr.Handle;
            var output = ret._blob_ptr.Handle;
            Util.call_check(NativeMethods.MXFuncInvoke(func_handle, ref input, new float[0], ref output));

            return ret;
        }


        [DebuggerHidden]
        public NDArrayHandle Get_handle() { return _blob_ptr.Handle; }

        public static NDArray Zeros(Shape shape, Context ctx =null, Type dtype=null)
        {
            if (ctx == null)
            {
                ctx = Context.Default_ctx;
            }
            if (dtype == null)
            {
                dtype = typeof(float);
            }

            var array = new NDArray(shape, ctx, false, dtype);
            array.Set_value(0);
            return array;
        }

        #region Numerics
        public SingleNArray As_numerics()
        {
            var shape = Get_shape();
            SingleNArray data = new SingleNArray(new numerics.nbase.Shape(shape.data()));
            var datagch = data.GetDataGcHandle();
            IntPtr pointer = datagch.AddrOfPinnedObject();
            var s = shape.Size();
            NativeMethods.MXNDArraySyncCopyToCPU(_blob_ptr.Handle, pointer, s);
            datagch.Free();
            return data;
        }
        #endregion


    }
}
