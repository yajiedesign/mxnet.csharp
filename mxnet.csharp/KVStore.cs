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
        private readonly KVStoreBlob _blob_ptr;
        private static KVStore _kvstore;
        private readonly string _kvtype;
        private MXKVStoreUpdater _updater_func;

        public KVStore(string name)
        {
            KVStoreHandle handle_;
            Util.call_check(NativeMethods.MXKVStoreCreate(name, out handle_));
            _kvtype = name;
            _blob_ptr = new KVStoreBlob(handle_);
        }


        public void Init(int key, NDArray val)
        {
            NDArrayHandle val_handle = val.Get_handle();
            Util.call_check(NativeMethods.MXKVStoreInit(_blob_ptr.Handle, 1, new int[] { key }, new NDArrayHandle[] { val_handle }));
        }

        public void Init(List<int> keys, List<NDArray> vals)
        {
            Util.assert(keys.Count == vals.Count);
            List<NDArrayHandle> val_handles = new List<NDArrayHandle>(vals.Count);
            val_handles.AddRange(vals.Select(s => s.Get_handle()));
            Util.call_check(NativeMethods.MXKVStoreInit(_blob_ptr.Handle, (uint)keys.Count, keys.ToArray(), val_handles.ToArray()));
        }

        public void Push(int key, NDArray val, int priority)
        {
            NDArrayHandle val_handle = val.Get_handle();
            Util.call_check(NativeMethods.MXKVStorePush(_blob_ptr.Handle, 1, new int[] { key }, new NDArrayHandle[] { val_handle }, priority));
        }

        public void Push(int key, List<NDArray> val, int priority)
        {
            var keys = Enumerable.Repeat(key, val.Count).ToList();
            Push(keys, val, priority);
        }

        public void Push(List<int> keys, List<NDArray> vals, int priority)
        {
            Util.assert(keys.Count == vals.Count);
            List<NDArrayHandle> val_handles = new List<NDArrayHandle>(vals.Count);
            val_handles.AddRange(vals.Select(s => s.Get_handle()));

            Util.call_check(NativeMethods.MXKVStorePush(_blob_ptr.Handle, (uint)keys.Count, keys.ToArray(), val_handles.ToArray(), priority));
        }

        public void Pull(int key, NDArray @out, int priority)
        {
            NDArrayHandle out_handle = @out.Get_handle();
            Util.call_check(NativeMethods.MXKVStorePull(_blob_ptr.Handle, 1, new[] { key }, new[] { out_handle }, priority));
        }

        public void Pull(int key, List<NDArray> outs, int priority)
        {
            var keys = Enumerable.Repeat(key, outs.Count).ToList();
            Pull(keys, outs, priority);
        }

        public void Pull(List<int> keys, List<NDArray> outs, int priority)
        {
            Util.assert(keys.Count == outs.Count);

            List<NDArrayHandle> out_handles = new List<NDArrayHandle>(keys.Count);
            out_handles.AddRange(outs.Select(s => s.Get_handle()));
            Util.call_check(NativeMethods.MXKVStorePull(_blob_ptr.Handle, (uint)keys.Count, keys.ToArray(), out_handles.ToArray(), priority));
        }



        public void set_optimizer(Optimizer optimizer)
        {


            int is_worker;
            Util.call_check(NativeMethods.MXKVStoreIsWorkerNode(out is_worker));


            if (_kvtype.Contains("dist") && is_worker != 0)
            {
                Util.call_check(NativeMethods.MXKVStoreSendCommmandToServers(_blob_ptr.Handle, 0, optimizer.Serialize()));
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

            Util.call_check(NativeMethods.MXKVStoreSetUpdater(_blob_ptr.Handle, this._updater_func, IntPtr.Zero));

        }

        public string Type
        {
            get
            {
                IntPtr type_ptr;
                Util.call_check(NativeMethods.MXKVStoreGetType(_blob_ptr.Handle, out type_ptr));
                // type is managed by handle_, no need to free its memory.
                return Marshal.PtrToStringAnsi(type_ptr);
            }
        }

        public int Get_rank()
        {
            int rank;
            Util.call_check(NativeMethods.MXKVStoreGetRank(_blob_ptr.Handle, out rank));
            return rank;
        }

        public int Get_num_workers()
        {
            int num_workers;
            Util.call_check(NativeMethods.MXKVStoreGetGroupSize(_blob_ptr.Handle, out num_workers));
            return num_workers;
        }

        public void Barrier()
        {
            Util.call_check(NativeMethods.MXKVStoreBarrier(_blob_ptr.Handle));
        }

        public string Get_role()
        {
            int ret;
            Util.call_check(NativeMethods.MXKVStoreIsSchedulerNode(out ret));
            if (ret != 0)
            {
                return "scheduler";
            }
            Util.call_check(NativeMethods.MXKVStoreIsServerNode(out ret));
            if (ret != 0)
            {
                return "server";
            }
            Util.call_check(NativeMethods.MXKVStoreIsWorkerNode(out ret));
            Util.assert(ret != 0);
            return "worker";
        }
    }
}
