using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Dynamic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Remoting.Contexts;
using System.Text;
using System.Threading.Tasks;
using mxnet.numerics.single;
using Microsoft.CSharp.RuntimeBinder;
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
    public partial class NdArray :INdArrayOrSymbol
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

        public NdArray(Shape shape, Context context, bool delayAlloc, Dtype dtype)
        {
            NDArrayHandle handle;
            Util.CallCheck(NativeMethods.MXNDArrayCreateEx(shape.Data().ToArray(), shape.Ndim(), context.DeviceType,
                               context.DeviceId, delayAlloc ? 1 : 0, dtype.Index, out handle));
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
            Util.CallCheck(NativeMethods.MXFuncInvoke(funcHandle, new [] { input}, new float[0], ref output));
            return other;
        }

        public NdArray CopyTo(Context ctx)
        {
            FunctionHandle funcHandle;
            NativeMethods.MXGetFunction("_copyto", out funcHandle);

            var other = new NdArray(this.GetShape(), ctx, true);
            var input = _blobPtr.Handle;
            var output = other._blobPtr.Handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(funcHandle, new [] { input}, new float[0], ref output));
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
            Util.CallCheck(NativeMethods.MXFuncInvoke(funcHandle, new[] { zero }, scalar, ref handle));
            return this;
        }

        public static void SampleGaussian(float mu, float sigma, NdArray outArray)
        {
            FunctionHandle funcHandle;
            NativeMethods.MXGetFunction("_random_gaussian", out funcHandle);
            float[] scalar = { mu, sigma };
            IntPtr zero = IntPtr.Zero;
            var handle = outArray._blobPtr.Handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(funcHandle, new[] { zero}, scalar, ref handle));
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

        public Dtype GetDtype()
        {
            int outDtype;
            Util.CallCheck(NativeMethods.MXNDArrayGetDType(_blobPtr.Handle, out outDtype));
            return (Dtype)outDtype;
        }


        public NdArray ArgmaxChannel()
        {
            NdArray ret = new NdArray();
            FunctionHandle funcHandle;
            NativeMethods.MXGetFunction("argmax_channel", out funcHandle);
            var input = _blobPtr.Handle;
            var output = ret._blobPtr.Handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(funcHandle, new[] { input}, new float[0], ref output));

            return ret;
        }


        [DebuggerHidden]
        public NDArrayHandle Handle => _blobPtr.Handle;

        //public static NdArray Zeros(Shape shape, Context ctx = null, Type dtype = null)
        //{
        //    if (ctx == null)
        //    {
        //        ctx = Context.DefaultCtx;
        //    }
        //    if (dtype == null)
        //    {
        //        dtype = typeof(float);
        //    }

        //    var array = new NdArray(shape, ctx, false, dtype);
        //    array.SetValue(0);
        //    return array;
        //}

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
            FunctionHandle funcHandle;
            NativeMethods.MXGetFunction("_plus", out funcHandle);
            var ret = new NdArray();
            var output = ret._blobPtr.Handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(funcHandle,  new NDArrayHandle[] { lhs.Handle,rhs.Handle }, new float[0], ref output));
            return ret;
        }

        public static NdArray operator -(NdArray lhs, NdArray rhs)
        {
            FunctionHandle funcHandle;
            NativeMethods.MXGetFunction("_minus", out funcHandle);
            var ret = new NdArray();
            var output = ret._blobPtr.Handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(funcHandle, new NDArrayHandle[] { lhs.Handle, rhs.Handle }, new float[0], ref output));
            return ret;
        }

        public static NdArray operator *(NdArray lhs, NdArray rhs)
        {
            FunctionHandle funcHandle;
            NativeMethods.MXGetFunction("_mul", out funcHandle);
            var ret = new NdArray();
            var output = ret._blobPtr.Handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(funcHandle, new NDArrayHandle[] { lhs.Handle, rhs.Handle }, new float[0], ref output));
            return ret;
        }

        public static NdArray operator /(NdArray lhs, NdArray rhs)
        {
            FunctionHandle funcHandle;
            NativeMethods.MXGetFunction("_div", out funcHandle);
            var ret = new NdArray();
            var output = ret._blobPtr.Handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(funcHandle, new NDArrayHandle[] { lhs.Handle, rhs.Handle }, new float[0], ref output));
            return ret;
        }

        public static NdArray operator +(NdArray lhs, float scalar)
        {
            FunctionHandle funcHandle;
            NativeMethods.MXGetFunction("_plus_scalar", out funcHandle);
            var ret = new NdArray();
            var output = ret._blobPtr.Handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(funcHandle, new NDArrayHandle[] { lhs.Handle,  }, new float[] { scalar }, ref output));
            return ret;
        }

        public static NdArray operator +(float scalar, NdArray rhs)
        {
            FunctionHandle funcHandle;
            NativeMethods.MXGetFunction("_plus_scalar", out funcHandle);
            var ret = new NdArray();
            var output = ret._blobPtr.Handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(funcHandle, new NDArrayHandle[] { rhs.Handle, }, new float[] { scalar }, ref output));
            return ret;
        }

        public static NdArray operator -(NdArray lhs, float scalar)
        {
            FunctionHandle funcHandle;
            NativeMethods.MXGetFunction("_minus_scalar", out funcHandle);
            var ret = new NdArray();
            var output = ret._blobPtr.Handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(funcHandle, new NDArrayHandle[] { lhs.Handle, }, new float[] { scalar }, ref output));
            return ret;
        }

        public static NdArray operator -(float scalar, NdArray rhs)
        {
            FunctionHandle funcHandle;
            NativeMethods.MXGetFunction("_rminus_scalar", out funcHandle);
            var ret = new NdArray();
            var output = ret._blobPtr.Handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(funcHandle, new NDArrayHandle[] { rhs.Handle, }, new float[] { scalar }, ref output));
            return ret;
        }

        public static NdArray operator *(NdArray lhs, float scalar)
        {
            FunctionHandle funcHandle;
            NativeMethods.MXGetFunction("_mul_scalar", out funcHandle);
            var ret = new NdArray();
            var output = ret._blobPtr.Handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(funcHandle, new NDArrayHandle[] { lhs.Handle, }, new float[] { scalar }, ref output));
            return ret;
        }

        public static NdArray operator *(float scalar, NdArray rhs)
        {
            FunctionHandle funcHandle;
            NativeMethods.MXGetFunction("_mul_scalar", out funcHandle);
            var ret = new NdArray();
            var output = ret._blobPtr.Handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(funcHandle, new NDArrayHandle[] { rhs.Handle, }, new float[] { scalar }, ref output));
            return ret;
        }

        public static NdArray operator /(NdArray lhs, float scalar)
        {
             
            //FunctionHandle funcHandle;
            //NativeMethods.NNGetOpHandle("_div_scalar", out funcHandle);
            var ret = new NdArray();
            //var output = ret._blobPtr.Handle;
            //Util.CallCheck(NativeMethods.MXImperativeInvoke(funcHandle, new NDArrayHandle[] { lhs.Handle, }, new float[] { scalar }, ref output));
            NdArrayDynamic.InInstance._div_scalar(lhs, scalar, @out: ret);

            return ret;
        }

        public static NdArray operator /(float scalar, NdArray rhs)
        {
            FunctionHandle funcHandle;
            NativeMethods.MXGetFunction("_rdiv_scalar", out funcHandle);
            var ret = new NdArray();
            var output = ret._blobPtr.Handle;
            Util.CallCheck(NativeMethods.MXFuncInvoke(funcHandle, new NDArrayHandle[] { rhs.Handle, }, new float[] { scalar }, ref output));
            return ret;
        }
        #endregion

        [DebuggerHidden]
        public NDArrayHandle get_handle()
        {
            return _blobPtr.Handle;
        }
    }

    public static class NdArrayExtension
    {
        public static NdArray Sum(this IEnumerable<NdArray> source)
        {
            return source.Aggregate((total, next) => total + next);
        }

    }

    public class NdArrayDynamic : DynamicObject
    {

        public static dynamic InInstance { get; } = new NdArrayDynamic();

        private Dictionary<string, NdarrayFunctionDelegate> _functions =
            new Dictionary<string, NdarrayFunctionDelegate>();

        public NdArrayDynamic()
        {
            InitNdarrayModule();
        }

        private void InitNdarrayModule()
        {

            uint opNamesSize;
            IntPtr opNamesArrayPtr;
            NativeMethods.NNListAllOpNames(out opNamesSize, out opNamesArrayPtr);
            IntPtr[] opNamesArrayPtrs = new IntPtr[opNamesSize];
            Marshal.Copy(opNamesArrayPtr, opNamesArrayPtrs, 0, (int)opNamesSize);
            IList<string> opNames = new List<string>();
            for (int i = 0; i < opNamesSize; i++)
            {
                opNames.Add(Marshal.PtrToStringAnsi(opNamesArrayPtrs[i]));
            }

            foreach (var opName in opNames)
            {
                FunctionHandle funcHandle;
                Util.NnCallCheck(NativeMethods.NNGetOpHandle(opName, out funcHandle));

                var function = _make_ndarray_function(funcHandle, opName);
                _functions.Add(opName, function);
            }


        }

        private delegate dynamic NdarrayFunctionDelegate(InvokeMemberBinder binder, object[] args);

        private NdarrayFunctionDelegate _make_ndarray_function(IntPtr funcHandle, string opName)
        {
            IntPtr namePtr;
            IntPtr descriptionPtr;
            uint numArgs = 0;
            IntPtr argNamesPtr;
            IntPtr argTypeInfosPtr;
            IntPtr argDescriptionsPtr;
            IntPtr keyVarNumArgsPtr;
            IntPtr returnTypePtr;
            Util.CallCheck( NativeMethods.MXSymbolGetAtomicSymbolInfo(funcHandle,
            out namePtr,
            out descriptionPtr,
            out numArgs,
            out argNamesPtr,
            out argTypeInfosPtr,
            out argDescriptionsPtr,
            out keyVarNumArgsPtr,
            out returnTypePtr));

            IntPtr[] argNamesPtrArray = new IntPtr[numArgs];
            IntPtr[] argTypeInfosPtrArray = new IntPtr[numArgs];
            IntPtr[] argDescriptionsPtrArray = new IntPtr[numArgs];

            if (numArgs > 0)
            {
                Marshal.Copy(argNamesPtr, argNamesPtrArray, 0, (int)numArgs);
                Marshal.Copy(argTypeInfosPtr, argTypeInfosPtrArray, 0, (int)numArgs);
                Marshal.Copy(argDescriptionsPtr, argDescriptionsPtrArray, 0, (int)numArgs);
            }
 
            List<string> arguments = new List<string>();

            for (int i = 0; i < numArgs; i++)
            {
                var dtype = Marshal.PtrToStringAnsi(argTypeInfosPtrArray[i]);
                if (dtype != null && !(dtype.StartsWith("NDArray") || dtype.StartsWith("Symbol")))
                {
                    arguments.Add(Marshal.PtrToStringAnsi(argNamesPtrArray[i]));
                }
            }


            return (InvokeMemberBinder binder, object[] args) => generic_ndarray_function(funcHandle, arguments, binder, args);


        }

        private dynamic generic_ndarray_function(IntPtr funcHandle, List<string> arguments, InvokeMemberBinder binder, object[] args)
        {
            var csharpBinder = binder.GetType().GetInterface("Microsoft.CSharp.RuntimeBinder.ICSharpInvokeOrInvokeMemberBinder");
            var argumentInfos = ((IList<CSharpArgumentInfo>) csharpBinder.GetProperty("ArgumentInfo").GetValue(binder, null)).Skip(1).ToList();
            var namedArgumentProperty = typeof(CSharpArgumentInfo).GetProperty("NamedArgument",
                System.Reflection.BindingFlags.NonPublic | 
                System.Reflection.BindingFlags.Instance  |
                System.Reflection.BindingFlags.GetProperty);

            var nametProperty = typeof(CSharpArgumentInfo).GetProperty("Name",
    System.Reflection.BindingFlags.NonPublic |
    System.Reflection.BindingFlags.Instance |
    System.Reflection.BindingFlags.GetProperty);

            Debug.Assert(argumentInfos != null, "argumentInfos != null");
            var nonamarg = argumentInfos.Where(w => !(bool)namedArgumentProperty.GetValue(w, null)).ToList();
            var namarg = argumentInfos.Select((x, i) => new { x, i }).Where(w => (bool)namedArgumentProperty.GetValue(w.x, null))
                .ToDictionary(k => nametProperty.GetValue(k.x, null) as string, v => args[v.i]).ToList();

            List<NDArrayHandle> nd_args = new List<NDArrayHandle>();
            List<NDArrayHandle> output_vars = new List<NDArrayHandle>();
            List<string> sparam_vals = new List<string>();
            List<string> sparam_keys = new List<string>();

            int pos_param_arg = 0;
            for (int i = 0; i < nonamarg.Count; i++)
            {
                if (args[i] is NdArray)
                {
                    nd_args.Add(((NdArray) args[i]).Handle);
                }
                else
                {
                    if (pos_param_arg >= arguments.Count)
                    {
                        throw new ArgumentException ("Too many positional arguments");
                    }

                    sparam_vals.Add(args[i].ToString());
                    sparam_keys.Add(arguments[pos_param_arg]);
                    pos_param_arg = pos_param_arg + 1;
                }
            }

            dynamic original_output = null;

            foreach (var kviem in namarg)
            {
                if (kviem.Key == "out")
                {
                    original_output = kviem.Value;
                    if (kviem.Value is NdArray)
                    {
                        output_vars.Add(((NdArray) kviem.Value).Handle);
                    }
                    else
                    {
                        foreach (var v in (IEnumerable) kviem.Value)
                        {
                            if (!(v is NdArray))
                            {
                                throw new ArgumentException("out need to be of type NDArray");
                            }
                            output_vars.Add(((NdArray) v).Handle);
                        }
                    }

                }
                else
                {
                    sparam_vals.Add(kviem.Value.ToString());
                    sparam_keys.Add(kviem.Key);
                }

            }
            int num_output = output_vars.Count;
            bool nooutput = num_output == 0;


            GCHandle? outputArrayGch = null;
            NDArrayHandle[] outputArray = null;
            IntPtr outputArrayPtr;

            if (nooutput)
            {
                outputArrayPtr = IntPtr.Zero;
            }
            else
            {
                outputArray = output_vars.ToArray();
                outputArrayGch = GCHandle.Alloc(outputArray, GCHandleType.Pinned);
                outputArrayPtr = outputArrayGch.Value.AddrOfPinnedObject();
            }


            NativeMethods.MXImperativeInvoke(funcHandle,
                nd_args.Count, nd_args.ToArray(),
                ref num_output, ref outputArrayPtr,
                sparam_keys.Count, sparam_keys.ToArray(), sparam_vals.ToArray());

            if (!nooutput)
            {
                outputArrayGch.Value.Free();
            }

            if (original_output != null)
            {
                return original_output;
            }
            if (nooutput)
            {
                NDArrayHandle[] ndArrays = new NDArrayHandle[num_output];
                Marshal.Copy(outputArrayPtr, ndArrays, 0, num_output);

                if (num_output == 1)
                {
                    return new NdArray(ndArrays[0]);
                }
                else
                {
                    return (IList<NdArray>) ndArrays.Select(s => new NdArray(s)).ToList();
                }

            }
            else
            {
                if (num_output == 1)
                {
                    return new NdArray(outputArray[0]);
                }
                else
                {
                    return (IList<NdArray>)outputArray.Select(s => new NdArray(s)).ToList();
                }
            }
        }

        public override bool TryInvokeMember(InvokeMemberBinder binder, object[] args, out object result)
        {

            if(_functions.ContainsKey(binder.Name))
            {
                result = _functions[binder.Name](binder, args);
                return true;
            }
            result = null;
            return false;
        }
   
    }
}
