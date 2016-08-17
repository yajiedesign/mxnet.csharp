using System;
using System.Collections.Generic;
using System.Linq;
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
            handle_ = handle;
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
            NativeMethods.MXSymbolFree(handle_);
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
        public SymbolHandle handle_ = IntPtr.Zero;

    }

    public class Symbol
    {
        SymBlob blob_ptr_;

        public Symbol(string name)
        {
            SymbolHandle symbolHandle;
            NativeMethods.MXSymbolCreateVariable(name, out symbolHandle);
            blob_ptr_ = new SymBlob(symbolHandle);
        }

        public Symbol(SymbolHandle symbolHandle)
        {
            blob_ptr_ = new SymBlob(symbolHandle);
        }

        public static Symbol Variable(string name)
        {
            return new Symbol(name);
        }


        public static Symbol operator +(Symbol lhs, Symbol rhs)
        {
            return Operator._Plus(lhs, rhs);
        }
        //public static Symbol operator -(Symbol rhs)
        //{
        //    return _Minus(this, rhs);
        //}
        //public static Symbol operator *(Symbol rhs) { return _Mul(this, rhs); }
        //public static Symbol operator /(Symbol rhs) { return _Div(this, rhs); }
        //public static Symbol operator +(float scalar)
        //{
        //    return _PlusScalar(this, scalar);
        //}
        //public static Symbol operator -(float scalar)
        //{
        //    return _MinusScalar(*this, scalar);
        //}
        //public static Symbol operator *(float scalar)
        //{
        //    return _MulScalar(*this, scalar);
        //}
        //public static Symbol operator /(float scalar)
        //{
        //    return _DivScalar(*this, scalar);
        //}


      public  SymbolHandle GetHandle() { return blob_ptr_.handle_; }
    }
}
