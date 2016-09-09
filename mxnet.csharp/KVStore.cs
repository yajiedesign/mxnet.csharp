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
    public class KvStoreBlob : IDisposable
    {

        /// <summary>
        ///  default constructor
        /// </summary>
        /// <summary>
        /// construct with SymbolHandle to store
        /// </summary>
        /// <param name="handle"></param>
        public KvStoreBlob(KVStoreHandle handle)

        {
            this.Handle = handle;
        }
        /// <summary>
        /// destructor, free the SymbolHandle
        /// </summary>
        ~KvStoreBlob()
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
    class KvStore
    {
        private readonly KvStoreBlob _blobPtr;
        private static KvStore _kvstore;
        private readonly string _kvtype;
        private MxkvStoreUpdater _updaterFunc;

        public KvStore(string name)
        {
            KVStoreHandle handle;
            Util.CallCheck(NativeMethods.MXKVStoreCreate(name, out handle));
            _kvtype = name;
            _blobPtr = new KvStoreBlob(handle);
        }


        public void Init(int key, NdArray val)
        {
            NDArrayHandle valHandle = val.Handle;
            Util.CallCheck(NativeMethods.MXKVStoreInit(_blobPtr.Handle, 1, new int[] { key }, new NDArrayHandle[] { valHandle }));
        }

        public void Init(List<int> keys, List<NdArray> vals)
        {
            Util.Assert(keys.Count == vals.Count);
            List<NDArrayHandle> valHandles = new List<NDArrayHandle>(vals.Count);
            valHandles.AddRange(vals.Select(s => s.Handle));
            Util.CallCheck(NativeMethods.MXKVStoreInit(_blobPtr.Handle, (uint)keys.Count, keys.ToArray(), valHandles.ToArray()));
        }

        public void Push(int key, NdArray val, int priority)
        {
            NDArrayHandle valHandle = val.Handle;
            Util.CallCheck(NativeMethods.MXKVStorePush(_blobPtr.Handle, 1, new int[] { key }, new NDArrayHandle[] { valHandle }, priority));
        }

        public void Push(int key, List<NdArray> val, int priority)
        {
            var keys = Enumerable.Repeat(key, val.Count).ToList();
            Push(keys, val, priority);
        }

        public void Push(List<int> keys, List<NdArray> vals, int priority)
        {
            Util.Assert(keys.Count == vals.Count);
            List<NDArrayHandle> valHandles = new List<NDArrayHandle>(vals.Count);
            valHandles.AddRange(vals.Select(s => s.Handle));

            Util.CallCheck(NativeMethods.MXKVStorePush(_blobPtr.Handle, (uint)keys.Count, keys.ToArray(), valHandles.ToArray(), priority));
        }

        public void Pull(int key, NdArray @out, int priority)
        {
            NDArrayHandle outHandle = @out.Handle;
            Util.CallCheck(NativeMethods.MXKVStorePull(_blobPtr.Handle, 1, new[] { key }, new[] { outHandle }, priority));
        }

        public void Pull(int key, List<NdArray> outs, int priority)
        {
            var keys = Enumerable.Repeat(key, outs.Count).ToList();
            Pull(keys, outs, priority);
        }

        public void Pull(List<int> keys, List<NdArray> outs, int priority)
        {
            Util.Assert(keys.Count == outs.Count);

            List<NDArrayHandle> outHandles = new List<NDArrayHandle>(keys.Count);
            outHandles.AddRange(outs.Select(s => s.Handle));
            Util.CallCheck(NativeMethods.MXKVStorePull(_blobPtr.Handle, (uint)keys.Count, keys.ToArray(), outHandles.ToArray(), priority));
        }



        public void SetOptimizer(Optimizer optimizer)
        {


            int isWorker;
            Util.CallCheck(NativeMethods.MXKVStoreIsWorkerNode(out isWorker));


            if (_kvtype.Contains("dist") && isWorker != 0)
            {
                Util.CallCheck(NativeMethods.MXKVStoreSendCommmandToServers(_blobPtr.Handle, 0, optimizer.Serialize()));
            }
            else
            {
                this._set_updater(Optimizer.GetUpdater(optimizer));
            }
        }

        private static MxkvStoreUpdater _updater_wrapper(Action<int, NdArray, NdArray> updater)
        {

            return (key, recv, local, handle) =>
            {
                var lhs = new NdArray(recv);
                var rhs = new NdArray(local);
                updater(key, lhs, rhs);
            };
        }

        private void _set_updater(Action<int, NdArray, NdArray> updater)
        {
            this._updaterFunc = _updater_wrapper(updater);

            Util.CallCheck(NativeMethods.MXKVStoreSetUpdater(_blobPtr.Handle, this._updaterFunc, IntPtr.Zero));

        }

        public string Type
        {
            get
            {
                IntPtr typePtr;
                Util.CallCheck(NativeMethods.MXKVStoreGetType(_blobPtr.Handle, out typePtr));
                // type is managed by handle_, no need to free its memory.
                return Marshal.PtrToStringAnsi(typePtr);
            }
        }

        public int GetRank()
        {
            int rank;
            Util.CallCheck(NativeMethods.MXKVStoreGetRank(_blobPtr.Handle, out rank));
            return rank;
        }

        public int GetNumWorkers()
        {
            int numWorkers;
            Util.CallCheck(NativeMethods.MXKVStoreGetGroupSize(_blobPtr.Handle, out numWorkers));
            return numWorkers;
        }

        public void Barrier()
        {
            Util.CallCheck(NativeMethods.MXKVStoreBarrier(_blobPtr.Handle));
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
