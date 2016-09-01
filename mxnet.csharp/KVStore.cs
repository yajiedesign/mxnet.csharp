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
            this.handle = handle;
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
            NativeMethods.MXKVStoreFree(handle);
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
        public KVStoreHandle handle { get; }
    }
    class KVStore
    {
        private readonly KVStoreBlob _blob_ptr;
        private static KVStore _kvstore;
        private readonly string _kvtype;
        private MXKVStoreUpdater _updater_func;

        public KVStore(string name)
        {
            KVStoreHandle handle_;
            Util.CallCheck(NativeMethods.MXKVStoreCreate(name, out handle_));
            _kvtype = name;
            _blob_ptr = new KVStoreBlob(handle_);
        }


        public void Init(int key, NDArray val)
        {
            NDArrayHandle val_handle = val.Get_handle();
            Util.CallCheck(NativeMethods.MXKVStoreInit(_blob_ptr.handle, 1, new int[] { key }, new NDArrayHandle[] { val_handle }));
        }

        public void Init(List<int> keys, List<NDArray> vals)
        {
            Util.Assert(keys.Count == vals.Count);
            List<NDArrayHandle> val_handles = new List<NDArrayHandle>(vals.Count);
            val_handles.AddRange(vals.Select(s => s.Get_handle()));
            Util.CallCheck(NativeMethods.MXKVStoreInit(_blob_ptr.handle, (uint)keys.Count, keys.ToArray(), val_handles.ToArray()));
        }

        public void Push(int key, NDArray val, int priority)
        {
            NDArrayHandle val_handle = val.Get_handle();
            Util.CallCheck(NativeMethods.MXKVStorePush(_blob_ptr.handle, 1, new int[] { key }, new NDArrayHandle[] { val_handle }, priority));
        }

        public void Push(int key, List<NDArray> val, int priority)
        {
            var keys = Enumerable.Repeat(key, val.Count).ToList();
            Push(keys, val, priority);
        }

        public void Push(List<int> keys, List<NDArray> vals, int priority)
        {
            Util.Assert(keys.Count == vals.Count);
            List<NDArrayHandle> val_handles = new List<NDArrayHandle>(vals.Count);
            val_handles.AddRange(vals.Select(s => s.Get_handle()));

            Util.CallCheck(NativeMethods.MXKVStorePush(_blob_ptr.handle, (uint)keys.Count, keys.ToArray(), val_handles.ToArray(), priority));
        }

        public void Pull(int key, NDArray @out, int priority)
        {
            NDArrayHandle out_handle = @out.Get_handle();
            Util.CallCheck(NativeMethods.MXKVStorePull(_blob_ptr.handle, 1, new[] { key }, new[] { out_handle }, priority));
        }

        public void Pull(int key, List<NDArray> outs, int priority)
        {
            var keys = Enumerable.Repeat(key, outs.Count).ToList();
            Pull(keys, outs, priority);
        }

        public void Pull(List<int> keys, List<NDArray> outs, int priority)
        {
            Util.Assert(keys.Count == outs.Count);

            List<NDArrayHandle> out_handles = new List<NDArrayHandle>(keys.Count);
            out_handles.AddRange(outs.Select(s => s.Get_handle()));
            Util.CallCheck(NativeMethods.MXKVStorePull(_blob_ptr.handle, (uint)keys.Count, keys.ToArray(), out_handles.ToArray(), priority));
        }



        public void set_optimizer(Optimizer optimizer)
        {


            int is_worker;
            Util.CallCheck(NativeMethods.MXKVStoreIsWorkerNode(out is_worker));


            if (_kvtype.Contains("dist") && is_worker != 0)
            {
                Util.CallCheck(NativeMethods.MXKVStoreSendCommmandToServers(_blob_ptr.handle, 0, optimizer.Serialize()));
            }
            else
            {
                this._set_updater(Optimizer.get_updater(optimizer));
            }
        }

        private static MXKVStoreUpdater _updater_wrapper(Action<int, NDArray, NDArray> updater)
        {

            return (key, recv, local, handle) =>
            {
                var lhs = new NDArray(recv);
                var rhs = new NDArray(local);
                updater(key, lhs, rhs);
            };
        }

        private void _set_updater(Action<int, NDArray, NDArray> updater)
        {
            this._updater_func = _updater_wrapper(updater);

            Util.CallCheck(NativeMethods.MXKVStoreSetUpdater(_blob_ptr.handle, this._updater_func, IntPtr.Zero));

        }

        public string type
        {
            get
            {
                IntPtr type_ptr;
                Util.CallCheck(NativeMethods.MXKVStoreGetType(_blob_ptr.handle, out type_ptr));
                // type is managed by handle_, no need to free its memory.
                return Marshal.PtrToStringAnsi(type_ptr);
            }
        }

        public int GetRank()
        {
            int rank;
            Util.CallCheck(NativeMethods.MXKVStoreGetRank(_blob_ptr.handle, out rank));
            return rank;
        }

        public int GetNumWorkers()
        {
            int num_workers;
            Util.CallCheck(NativeMethods.MXKVStoreGetGroupSize(_blob_ptr.handle, out num_workers));
            return num_workers;
        }

        public void Barrier()
        {
            Util.CallCheck(NativeMethods.MXKVStoreBarrier(_blob_ptr.handle));
        }

        public string GetRole()
        {
            int ret;
            Util.CallCheck(NativeMethods.MXKVStoreIsSchedulerNode(out ret));
            if (ret != 0)
            {
                return "scheduler";
            }
            Util.CallCheck(NativeMethods.MXKVStoreIsServerNode(out ret));
            if (ret != 0)
            {
                return "server";
            }
            Util.CallCheck(NativeMethods.MXKVStoreIsWorkerNode(out ret));
            Util.Assert(ret != 0);
            return "worker";
        }
    }
}
