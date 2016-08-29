using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using mxnet.csharp.optimizer;
using KVStoreHandle = System.IntPtr;
using NDArrayHandle = System.IntPtr;

namespace mxnet.csharp
{
    public class KVStoreBlob : IDisposable
    {

        /// <summary>
        ///  default constructor
        /// </summary>
        /// <summary>
        /// construct with SymbolHandle to store
        /// </summary>
        /// <param name="handle"></param>
        public KVStoreBlob(KVStoreHandle handle)

        {
            Handle = handle;
        }
        /// <summary>
        /// destructor, free the SymbolHandle
        /// </summary>
        ~KVStoreBlob()
        {
            Dispose(false);
        }

        private void Dispose(bool disposing)
        {
            NativeMethods.MXKVStoreFree(Handle);
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
        public KVStoreHandle Handle { get; }
    }
    class KVStore
    {
        private KVStoreBlob _blobPtr;
        private static KVStore kvstore;
        private string _kvtype;
        private MXKVStoreUpdater _updater_func;

        public KVStore(string name)
        {
            KVStoreHandle handle_;
            Debug.Assert(NativeMethods. MXKVStoreCreate(name, out handle_)== 0);
            _kvtype = name;

            _blobPtr = new KVStoreBlob(handle_);
        }


        public void Init(int key, NDArray val)
        {
            NDArrayHandle val_handle = val.GetHandle();
            Debug.Assert(NativeMethods. MXKVStoreInit(_blobPtr.Handle, 1, new int[] { key }, new NDArrayHandle[] { val_handle }) == 0);
        }

        public void Init(List<int> keys, List<NDArray> vals)
        {
            Debug.Assert(keys.Count== vals.Count);
            List<NDArrayHandle> val_handles = new List<NDArrayHandle>(vals.Count);
            val_handles.AddRange(vals.Select(s => s.GetHandle()));
            Debug.Assert(NativeMethods.MXKVStoreInit(_blobPtr.Handle, (uint)keys.Count, keys.ToArray(), val_handles.ToArray())== 0);
        }

        public void Push(int key, NDArray val, int priority)
        {
            NDArrayHandle val_handle = val.GetHandle();
            Debug.Assert(NativeMethods.MXKVStorePush(_blobPtr.Handle, 1, new int[] { key }, new NDArrayHandle[] { val_handle }, priority) == 0);
        }

        public void Push(int key, List<NDArray> val, int priority)
        {
            var keys = Enumerable.Repeat(key, val.Count).ToList();
            Push(keys, val, priority);
        }

        public void Push(List<int> keys, List<NDArray> vals, int priority)
        {
            Debug.Assert(keys.Count== vals.Count);
            List<NDArrayHandle> val_handles = new List<NDArrayHandle>(vals.Count);
            val_handles.AddRange(vals.Select(s => s.GetHandle()));

            Debug.Assert(NativeMethods.MXKVStorePush(_blobPtr.Handle, (uint)keys.Count, keys.ToArray(), val_handles.ToArray(), priority)== 0);
        }

        public void Pull(int key, NDArray @out, int priority)
        {
            NDArrayHandle out_handle = @out.GetHandle();
            Debug.Assert(NativeMethods.MXKVStorePull(_blobPtr.Handle, 1,new[] { key}, new[] { out_handle}, priority)== 0);
        }

        public void Pull(int key, List<NDArray> outs, int priority)
        {
            var keys = Enumerable.Repeat(key, outs.Count).ToList();
            Pull(keys, outs, priority);
        }

        public void Pull(List<int> keys, List<NDArray> outs, int priority)
        {
            Debug.Assert(keys.Count == outs.Count);

            List<NDArrayHandle> out_handles = new List<NDArrayHandle>(keys.Count);
            out_handles.AddRange(outs.Select(s => s.GetHandle()));
            Debug.Assert(NativeMethods.MXKVStorePull(_blobPtr.Handle, (uint)keys.Count, keys.ToArray(), out_handles.ToArray(), priority) == 0);
        }



        public void set_optimizer(Optimizer optimizer)
        {


            int is_worker;
            Debug.Assert(NativeMethods.MXKVStoreIsWorkerNode(out is_worker) == 0);


            if (_kvtype.Contains("dist") && is_worker!=0)
            {
                Debug.Assert(NativeMethods.MXKVStoreSendCommmandToServers(_blobPtr.Handle, 0, optimizer.Serialize()) == 0);
            }
            else
            {
                this._set_updater(Optimizer.get_updater(optimizer));
            }
        }

        private static MXKVStoreUpdater _updater_wrapper(Action<int, NDArray, NDArray> updater)
        {

            return (int key, System.IntPtr recv, System.IntPtr local, System.IntPtr handle) =>
            {
                var lhs =new  NDArray(recv);
                var rhs = new  NDArray(local);
                updater(key, lhs, rhs);
            };
        }

        private void _set_updater(Action<int, NDArray, NDArray> updater)
        {
            this._updater_func = _updater_wrapper(updater);

            Debug.Assert(NativeMethods.MXKVStoreSetUpdater(_blobPtr.Handle, this._updater_func, IntPtr.Zero) == 0);

        }

        public string Type
        {
            get
            {
                IntPtr type_ptr;
                Debug.Assert(NativeMethods.MXKVStoreGetType(_blobPtr.Handle, out type_ptr) == 0);
                // type is managed by handle_, no need to free its memory.
                return Marshal.PtrToStringAnsi(type_ptr);
            }
        }

        public int GetRank()
        {
            int rank;
            Debug.Assert(NativeMethods.MXKVStoreGetRank(_blobPtr.Handle, out rank)== 0);
            return rank;
        }

        public int GetNumWorkers()
        {
            int num_workers;
            Debug.Assert(NativeMethods.MXKVStoreGetGroupSize(_blobPtr.Handle, out num_workers)== 0);
            return num_workers;
        }

        public void Barrier()
        {
            Debug.Assert(NativeMethods.MXKVStoreBarrier(_blobPtr.Handle)== 0);
        }

        public string GetRole()
        {
            int ret;
            Debug.Assert(NativeMethods.MXKVStoreIsSchedulerNode(out ret) == 0);
            if (ret != 0)
            {
                return "scheduler";
            }
            Debug.Assert(NativeMethods.MXKVStoreIsServerNode(out ret) == 0);
            if (ret != 0)
            {
                return "server";
            }
            Debug.Assert(NativeMethods.MXKVStoreIsWorkerNode(out ret) == 0);
            Debug.Assert(ret != 0);
            return "worker";
        }
    }
}
