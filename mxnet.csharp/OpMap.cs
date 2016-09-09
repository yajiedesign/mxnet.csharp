using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using AtomicSymbolCreator = System.IntPtr;

namespace mxnet.csharp
{
    class OpMap
    {
        /// <summary>
        ///  Create an Mxnet instance
        /// </summary>
        public OpMap()
        {
            uint numSymbolCreators = 0;
            IntPtr symbolCreatorsPtr;
     
            int r =  NativeMethods.MXSymbolListAtomicSymbolCreators(out numSymbolCreators,out symbolCreatorsPtr);
            Util.Assert(r == 0);
            AtomicSymbolCreator[] symbolCreators = new AtomicSymbolCreator[numSymbolCreators];
            Marshal.Copy(symbolCreatorsPtr, symbolCreators, 0, (int)numSymbolCreators);

            for (int i = 0; i < numSymbolCreators; i++)
            {
                IntPtr namePtr;
                IntPtr descriptionPtr;
                uint numArgs = 0;
                IntPtr argNamesPtr;
                IntPtr argTypeInfosPtr;
                IntPtr argDescriptionsPtr;
                IntPtr keyVarNumArgsPtr;
                IntPtr returnTypePtr;
                r = NativeMethods.MXSymbolGetAtomicSymbolInfo(symbolCreators[i],
                out namePtr,
                out descriptionPtr,
                out numArgs,
                out argNamesPtr,
                out argTypeInfosPtr,
                out argDescriptionsPtr,
                out keyVarNumArgsPtr,
                out returnTypePtr);
                Util.Assert(r == 0);

                string name = Marshal.PtrToStringAnsi(namePtr);
                _symbolCreators[name] = symbolCreators[i];
            }
        }

        /*!
        * \brief Get a symbol creator with its name.
        *
        * \param name name of the symbol creator
        * \return handle to the symbol creator
        */
        public AtomicSymbolCreator GetSymbolCreator(string name)
        {
            return _symbolCreators[name];
        }


        readonly Dictionary<string, AtomicSymbolCreator> _symbolCreators = new Dictionary<string, AtomicSymbolCreator>();
    }
}
