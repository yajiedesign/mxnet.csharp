using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using SymbolHandle = System.IntPtr;

namespace mxnet.csharp
{
    public class SymBlob : IDisposable
    {

        /// <summary>
        ///  default constructor
        /// </summary>
        SymBlob()
        {
        }
        /// <summary>
        /// construct with SymbolHandle to store
        /// </summary>
        /// <param name="handle"></param>
        public SymBlob(SymbolHandle handle)

        {
            Handle = handle;
        }
        /// <summary>
        /// destructor, free the SymbolHandle
        /// </summary>
        ~SymBlob()
        {
            Dispose(false);
        }

        private void Dispose(bool disposing)
        {
            NativeMethods.MXSymbolFree(Handle);
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
        public SymbolHandle Handle = IntPtr.Zero;

    }

    public class Symbol : OperatorWarp
    {
        readonly SymBlob _blobPtr;

        public Symbol(string name)
        {
            SymbolHandle symbolHandle;
            NativeMethods.MXSymbolCreateVariable(name, out symbolHandle);
            _blobPtr = new SymBlob(symbolHandle);
        }

        public Symbol(SymbolHandle symbolHandle)
        {
            _blobPtr = new SymBlob(symbolHandle);
        }

        public static Symbol Variable(string name)
        {
            return new Symbol(name);
        }


        public static Symbol operator +(Symbol lhs, Symbol rhs)
        {
            return Operator._Plus(lhs, rhs);
        }
        public static Symbol operator -(Symbol lhs, Symbol rhs)
        {
            return Operator._Minus(lhs, rhs);
        }
        public static Symbol operator *(Symbol lhs, Symbol rhs) { return Operator._Mul(lhs, rhs); }
        public static Symbol operator /(Symbol lhs, Symbol rhs) { return Operator._Div(lhs, rhs); }
        public static Symbol operator +(Symbol lhs, float scalar)
        {
            return Operator._PlusScalar(lhs, scalar);
        }
        public static Symbol operator +(float scalar, Symbol rhs)
        {
            return Operator._PlusScalar(rhs, scalar);
        }
        public static Symbol operator -(Symbol lhs, float scalar)
        {
            return Operator._MinusScalar(lhs, scalar);
        }
        public static Symbol operator -(float scalar, Symbol rhs)
        {
            return Operator._RMinusScalar(scalar, rhs);
        }
        public static Symbol operator *(Symbol lhs, float scalar)
        {
            return Operator._MulScalar(lhs, scalar);
        }
        public static Symbol operator *(float scalar, Symbol rhs)
        {
            return Operator._MulScalar(rhs, scalar);
        }
        public static Symbol operator /(Symbol lhs, float scalar)
        {
            return Operator._DivScalar(lhs, scalar);
        }
        public static Symbol operator /(float scalar, Symbol rhs)
        {
            return Operator._RDivScalar(scalar, rhs);
        }


        public Symbol this[int index]
        {
            get
            {
                SymbolHandle @out;
                NativeMethods.MXSymbolGetOutput(GetHandle(), (uint)index, out @out);
                return new Symbol(@out);
            }
        }
        public Symbol this[string index]
        {
            get
            {
                var outputs = ListOutputs();
                for (int i = 0; i < outputs.Count; i++)
                {
                    if (outputs[i] == index)
                    {
                        return this[i];
                    }
                }
                return this[0];
            }
        }

        public Symbol Group(IEnumerable<Symbol> symbols)
        {

            List<SymbolHandle> handle_list = new List<SymbolHandle>();
            foreach (var symbol in symbols)
            {
                handle_list.Add(symbol.GetHandle());
            }
            SymbolHandle @out;

            NativeMethods.MXSymbolCreateGroup((uint)handle_list.Count, handle_list.ToArray(), out @out);
            return new Symbol(@out);
        }
        public Symbol Load(string file_name)
        {
            SymbolHandle handle;
            Debug.Assert(NativeMethods.MXSymbolCreateFromFile(file_name, out handle) == 0);
            return new Symbol(handle);
        }
        public Symbol LoadJSON(string json_str)
        {
            SymbolHandle handle;
            Debug.Assert(NativeMethods.MXSymbolCreateFromJSON(json_str, out handle) == 0);
            return new Symbol(handle);
        }

        void Save(string file_name)
        {
            Debug.Assert(NativeMethods.MXSymbolSaveToFile(GetHandle(), file_name) == 0);
        }
        public string ToJSON()
        {
            IntPtr out_json;
            Debug.Assert(NativeMethods.MXSymbolSaveToJSON(GetHandle(), out out_json) == 0);
            return Marshal.PtrToStringAnsi(out_json);
        }
        public Symbol GetInternals()
        {
            SymbolHandle handle;
            Debug.Assert(NativeMethods.MXSymbolGetInternals(GetHandle(), out handle) == 0);
            return new Symbol(handle);
        }

        public Symbol Copy() 
        {
            SymbolHandle handle;
            Debug.Assert(NativeMethods.MXSymbolCopy(GetHandle(), out handle) == 0);
            return new Symbol(handle);
        }
        public IList<string> ListArguments()
        {
            List<string> ret = new List<string>();
            uint size;
            IntPtr sarrPtr;

            NativeMethods.MXSymbolListArguments(GetHandle(), out size, out sarrPtr);
            IntPtr[] sarr = new IntPtr[size];
            if (size > 0)
            {
                Marshal.Copy(sarrPtr, sarr, 0, (int)size);
            }
            for (int i = 0; i < size; i++)
            {
                ret.Add(Marshal.PtrToStringAnsi(sarr[i]));
            }
            return ret;
        }
        public IList<string> ListOutputs()
        {
            List<string> ret = new List<string>();
            uint size;
            IntPtr sarrPtr;

            NativeMethods.MXSymbolListOutputs(GetHandle(), out size, out sarrPtr);
            IntPtr[] sarr = new IntPtr[size];
            if (size > 0)
            {
                Marshal.Copy(sarrPtr, sarr, 0, (int)size);
            }
            for (int i = 0; i < size; i++)
            {
                ret.Add(Marshal.PtrToStringAnsi(sarr[i]));
            }
            return ret;
        }
        public IList<string> ListAuxiliaryStates()
        {
            List<string> ret = new List<string>();
            uint size;
            IntPtr sarrPtr;

            NativeMethods.MXSymbolListAuxiliaryStates(GetHandle(), out size, out sarrPtr);
            IntPtr[] sarr = new IntPtr[size];
            if (size > 0)
            {
                Marshal.Copy(sarrPtr, sarr, 0, (int)size);
            }
            for (int i = 0; i < size; i++)
            {
                ret.Add(Marshal.PtrToStringAnsi(sarr[i]));
            }
            return ret;
        }


        public SymbolHandle GetHandle() { return _blobPtr.Handle; }
    }
}
