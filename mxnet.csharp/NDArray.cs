using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Remoting.Contexts;
using System.Text;
using System.Threading.Tasks;
using NDArrayHandle = System.IntPtr;

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
        ///  default constructor
        /// </summary>
        NDBlob()
        {
        }
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
        NDBlob _blobPtr;
        public NDArray()
        {
            NDArrayHandle handle;
            Debug.Assert(NativeMethods.MXNDArrayCreateNone(out handle) == 0);
            _blobPtr = new NDBlob(handle);
        }
        public NDArray(NDArrayHandle handle)
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
        public NDArray(Shape shape, Context context, bool delayAlloc)
        {
            NDArrayHandle handle;
            Debug.Assert(NativeMethods.MXNDArrayCreate(shape.data().ToArray(), shape.ndim(), context.GetDeviceType(),
                               context.GetDeviceId(), delayAlloc ? 1 : 0, out handle) == 0);
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


        public uint Size()
        {
            uint ret = 1;
            foreach (var i in GetShape()) { ret *= (uint)i; }
            return ret;
        }

        public uint[] GetShape()
        {
            IntPtr outPdata;
            uint outDim;
            NativeMethods.MXNDArrayGetShape(_blobPtr.Handle, out outDim, out outPdata);
            int[] ret = new int[outDim];
            Marshal.Copy(outPdata, ret, 0, (int)outDim);
            return ret.Select(s => (uint)s).ToArray();
        }

        public NDArrayHandle GetHandle() { return _blobPtr.Handle; }
    }
}
