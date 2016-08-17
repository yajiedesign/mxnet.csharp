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
            uint num_symbol_creators = 0;
            IntPtr symbol_creators_ptr = IntPtr.Zero;
     
            int r =  NativeMethods.MXSymbolListAtomicSymbolCreators(out num_symbol_creators,out symbol_creators_ptr);
            Debug.Assert(r == 0);
            AtomicSymbolCreator[] symbol_creators = new AtomicSymbolCreator[num_symbol_creators];
            Marshal.Copy(symbol_creators_ptr, symbol_creators, 0, (int)num_symbol_creators);

            for (int i = 0; i < num_symbol_creators; i++)
            {
                IntPtr name_ptr = IntPtr.Zero;
                IntPtr description_ptr = IntPtr.Zero;
                uint num_args = 0;
                IntPtr arg_names_ptr = IntPtr.Zero;
                IntPtr arg_type_infos_ptr = IntPtr.Zero;
                IntPtr arg_descriptions_ptr = IntPtr.Zero;
                IntPtr key_var_num_args_ptr = IntPtr.Zero;
                IntPtr return_type_ptr = IntPtr.Zero;
                r = NativeMethods.MXSymbolGetAtomicSymbolInfo(symbol_creators[i],
                out name_ptr,
                out description_ptr,
                out num_args,
                out arg_names_ptr,
                out arg_type_infos_ptr,
                out arg_descriptions_ptr,
                out key_var_num_args_ptr,
                out return_type_ptr);
                Debug.Assert(r == 0);

                string name = Marshal.PtrToStringAnsi(name_ptr);
                symbol_creators_[name] = symbol_creators[i];
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
            return symbol_creators_[name];
        }


        Dictionary<string, AtomicSymbolCreator> symbol_creators_ = new Dictionary<string, AtomicSymbolCreator>();
    }
}
