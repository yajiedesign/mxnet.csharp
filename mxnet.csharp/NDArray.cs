using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Remoting.Contexts;
using System.Text;
using System.Threading.Tasks;
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
        /// <summary>
        /// 
        /// </summary>
        /// <returns>the type of the device</returns>
        public DeviceType GetDeviceType() { return _type; }

        /// <summary>
        /// 
        /// </summary>
        /// <returns>the id of the device</returns>
        public int GetDeviceId() { return _id; }


        /// <summary>
        /// Return a GPU context
        /// </summary>
        /// <param name="deviceId">id of the device</param>
        /// <returns>the corresponding GPU context</returns>
        public static Context Gpu(int deviceId = 0)
        {
            return new Context(DeviceType.KGpu, deviceId);
        }


        /// <summary>
        /// Return a CPU context
        /// </summary>
        /// <param name="deviceId">id of the device. this is not needed by CPU</param>
        /// <returns>the corresponding CPU context</returns>
        public static Context Cpu(int deviceId = 0)
        {
            return new Context(DeviceType.KCpu, deviceId);
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
        public NDArrayHandle Handle { get; } = IntPtr.Zero;
    }
    public class NDArray
    {
        private readonly NDBlob _blobPtr;
        public NDArray()
        {
            NDArrayHandle handle;
            Debug.Assert(NativeMethods.MXNDArrayCreateNone(out handle) == 0);
            _blobPtr = new NDBlob(handle);
        }
        public NDArray(NDArrayHandle handle,bool writable=true)
        {
            _blobPtr = new NDBlob(handle);
        }
        public NDArray(uint[] shape, Context context,
                        bool delayAlloc)
        {
            NDArrayHandle handle;
            Debug.Assert(NativeMethods.MXNDArrayCreate(shape, (uint)shape.Length, context.GetDeviceType(),
                           context.GetDeviceId(), delayAlloc ? 1 : 0, out handle) == 0);
            _blobPtr = new NDBlob(handle);
        }
        public NDArray(uint[] shape)
        {
            NDArrayHandle handle;
            Debug.Assert(NativeMethods.MXNDArrayCreate(shape, (uint)shape.Length,  DeviceType.KCpu,  0, 0, out handle) == 0);
            _blobPtr = new NDBlob(handle);
        }
        public NDArray(Shape shape, Context context, bool delayAlloc)
        {
            NDArrayHandle handle;
            Debug.Assert(NativeMethods.MXNDArrayCreate(shape.data().ToArray(), shape.ndim(), context.GetDeviceType(),
                               context.GetDeviceId(), delayAlloc ? 1 : 0, out handle) == 0);
            _blobPtr = new NDBlob(handle);
        }

        public NDArray(Shape shape, Context context, bool delayAlloc ,Type dtype)
        {
            NDArrayHandle handle;
            Debug.Assert(NativeMethods.MXNDArrayCreateEx(shape.data().ToArray(), shape.ndim(), context.GetDeviceType(),
                               context.GetDeviceId(), delayAlloc ? 1 : 0, Util._DTYPE_NP_TO_MX[dtype],out handle) == 0);
            _blobPtr = new NDBlob(handle);
        }

        public NDArray(float[] data)
        {
            NDArrayHandle handle;
            Debug.Assert(NativeMethods.MXNDArrayCreateNone(out handle) == 0);
            NativeMethods.MXNDArraySyncCopyFromCPU(handle, data, (uint)data.Length);
            _blobPtr = new NDBlob(handle);
        }
        public NDArray(float[] data, Shape shape,
                         Context context)
        {
            NDArrayHandle handle;
            Debug.Assert(NativeMethods.MXNDArrayCreate(shape.data().ToArray(), shape.ndim(), context.GetDeviceType(),
                           context.GetDeviceId(), 0, out handle) == 0);
            NativeMethods.MXNDArraySyncCopyFromCPU(handle, data, shape.Size());
            _blobPtr = new NDBlob(handle);
        }


        public void SyncCopyFromCPU(float[] data)
        {
            NativeMethods.MXNDArraySyncCopyFromCPU(_blobPtr.Handle, data, (uint) data.Length);
        }

        public float[] SyncCopyToCPU(uint size)
        {
            size = size > 0 ? size : Size();
            var data = new float[size];
            var datagch = GCHandle.Alloc(data);
            //  var dataPtr = Marshal.AllocHGlobal(sizeof(float) * (int)size);
            NativeMethods.MXNDArraySyncCopyToCPU(_blobPtr.Handle, (IntPtr)datagch, size);
            // Marshal.Copy(dataPtr, data, 0, (int)size);
            // Marshal.FreeHGlobal(dataPtr);
            datagch.Free();
            return data;
        }

        public void WaitToRead()
        {
            Debug.Assert(NativeMethods.MXNDArrayWaitToRead(_blobPtr.Handle) == 0);
        }
        public void WaitToWrite()
        {
            Debug.Assert(NativeMethods.MXNDArrayWaitToWrite(_blobPtr.Handle) == 0);
        }
        public static void WaitAll() { Debug.Assert(NativeMethods.MXNDArrayWaitAll() == 0); }

        public NDArray CopyTo(NDArray other)
        {
            FunctionHandle func_handle;
            NativeMethods.MXGetFunction("_copyto", out func_handle);

            var input = _blobPtr.Handle;
            var output = other._blobPtr.Handle;
            Debug.Assert(NativeMethods.MXFuncInvoke(func_handle, ref input, new float[0], ref output) == 0);
            return other;
        }

        public NDArray Slice(uint begin, uint end)
        {
            NDArrayHandle handle;
            Debug.Assert(NativeMethods.MXNDArraySlice(GetHandle(), begin, end, out handle) == 0);
            return new NDArray(handle);
        }

        public NDArray Reshape(Shape new_shape)
        {
            NDArrayHandle handle;
            var dims = new_shape.data().Select(s => (int)s );
            Debug.Assert(NativeMethods.MXNDArrayReshape(GetHandle(), (int)new_shape.ndim(), dims.ToArray(),out handle)==0);
            return new NDArray(handle);
        }

        public NDArray SetValue(float value)
        {
            FunctionHandle func_handle;
            NativeMethods.MXGetFunction("_set_value", out func_handle);
            float[] scalar = { value };
            IntPtr Zero = IntPtr.Zero;
            var handle = _blobPtr.Handle;
            Debug.Assert(NativeMethods.MXFuncInvoke(func_handle, ref Zero, scalar, ref handle) == 0);
            return this;
        }

        public static  void SampleGaussian(float mu, float sigma, NDArray outArray)
        {
            FunctionHandle func_handle;
            NativeMethods.MXGetFunction("_random_gaussian", out func_handle);
            float[] scalar = { mu, sigma };
            IntPtr Zero = IntPtr.Zero;
            var handle = outArray._blobPtr.Handle;
            Debug.Assert(NativeMethods.MXFuncInvoke(func_handle, ref Zero, scalar, ref handle) == 0);
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
            int out_dtype;
            Debug.Assert(NativeMethods.MXNDArrayGetDType(_blobPtr.Handle, out out_dtype) == 0);
            return Util._DTYPE_MX_TO_NP[out_dtype];
        }


        public NDArrayHandle GetHandle() { return _blobPtr.Handle; }

        public static NDArray Zeros(Shape shape, Context ctx, Type dtype)
        {
            var array = new NDArray(shape, ctx, false, dtype);
            array.SetValue(0);
            return array;
        }
    }
}
