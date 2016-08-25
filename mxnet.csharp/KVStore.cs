using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
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
        private Optimizer optimizer_;
        private static KVStore kvstore;

        public KVStore(string name)
        {
            KVStoreHandle handle_;
            Debug.Assert(NativeMethods. MXKVStoreCreate(name, out handle_)== 0);

            _blobPtr = new KVStoreBlob(handle_);
        }

        public KVStore(KVStore kv)
        {
            optimizer_ = kv.optimizer_;
            _blobPtr = kv._blobPtr;
        }

        public void RunServer()
        {
            Debug.Assert(GetRole()!= "worker");
            kvstore = this;
            Debug.Assert(NativeMethods.MXKVStoreRunServer(_blobPtr.Handle, controller, IntPtr.Zero)== 0);
        }

        private void controller(int head, string body, IntPtr controller_handle)
        {

            if (kvstore == null)
            {
                return;
            }
            if (head == 0)
            {
                SortedDictionary<string, string> @params = new SortedDictionary<string, string>();
                var lines = body.Split('\n');
                foreach (var line in lines)
                {
                    var sp = line.Split('=');
                    @params.Add(sp[0], sp[1]);
                }


                float lr = Convert.ToSingle(@params["learning_rate"]);
                float wd = Convert.ToSingle(@params["weight_decay"]);
                var opt = new Optimizer(@params["opt_type"], lr, wd);
                @params.Remove("opt_type");
                @params.Remove("learning_rate");
                @params.Remove("weight_decay");
                foreach (var pair in @params)
                {
                    opt.SetParam(pair.Key, pair.Value);
                }
                kvstore.SetOptimizer(opt, true);

            }

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

        public void Pull(List<int> keys, List<NDArray> outs, int priority)
        {
            Debug.Assert(keys.Count== outs.Count);

            List<NDArrayHandle> out_handles = new List<NDArrayHandle>(keys.Count);
            out_handles.AddRange(outs.Select(s => s.GetHandle()));

            Debug.Assert(NativeMethods.MXKVStorePull(_blobPtr.Handle, (uint)keys.Count, keys.ToArray(), out_handles.ToArray(), priority) == 0);
          }

        private void updater(int key, NDArrayHandle recv, NDArrayHandle local, IntPtr handle_)
        {
            GCHandle optgch = GCHandle.FromIntPtr(handle_);
            Optimizer opt = (Optimizer) optgch.Target;

            opt.Update(key, new NDArray(local), new NDArray(recv));
        }

        public void SetOptimizer(Optimizer optimizer, bool local)
        {
            if (local)
            {
                optimizer_ = optimizer;

                var optimizergch = GCHandle.Alloc(optimizer_);
                Debug.Assert(NativeMethods.MXKVStoreSetUpdater(_blobPtr.Handle, updater, (IntPtr)optimizergch) == 0);
                optimizergch.Free();
            }
            else
            {
                Debug.Assert(NativeMethods.MXKVStoreSendCommmandToServers(_blobPtr.Handle, 0, optimizer.Serialize())== 0);
            }
        }

        public string GetType()
        {
            IntPtr type_ptr;
            Debug.Assert(NativeMethods.MXKVStoreGetType(_blobPtr.Handle, out type_ptr)== 0);
            // type is managed by handle_, no need to free its memory.
            return Marshal.PtrToStringAnsi(type_ptr);
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
