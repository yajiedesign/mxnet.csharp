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
            IntPtr symbol_creators_ptr;
     
            int r =  NativeMethods.MXSymbolListAtomicSymbolCreators(out num_symbol_creators,out symbol_creators_ptr);
            Util.assert(r == 0);
            AtomicSymbolCreator[] symbol_creators = new AtomicSymbolCreator[num_symbol_creators];
            Marshal.Copy(symbol_creators_ptr, symbol_creators, 0, (int)num_symbol_creators);

            for (int i = 0; i < num_symbol_creators; i++)
            {
                IntPtr name_ptr;
                IntPtr description_ptr;
                uint num_args = 0;
                IntPtr arg_names_ptr;
                IntPtr arg_type_infos_ptr;
                IntPtr arg_descriptions_ptr;
                IntPtr key_var_num_args_ptr;
                IntPtr return_type_ptr;
                r = NativeMethods.MXSymbolGetAtomicSymbolInfo(symbol_creators[i],
                out name_ptr,
                out description_ptr,
                out num_args,
                out arg_names_ptr,
                out arg_type_infos_ptr,
                out arg_descriptions_ptr,
                out key_var_num_args_ptr,
                out return_type_ptr);
                Util.assert(r == 0);

                string name = Marshal.PtrToStringAnsi(name_ptr);
                _symbol_creators_[name] = symbol_creators[i];
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
            return _symbol_creators_[name];
        }


        readonly Dictionary<string, AtomicSymbolCreator> _symbol_creators_ = new Dictionary<string, AtomicSymbolCreator>();
    }
}
