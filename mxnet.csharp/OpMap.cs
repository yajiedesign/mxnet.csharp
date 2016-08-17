using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
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
            AtomicSymbolCreator[] symbol_creators = new AtomicSymbolCreator[0];
            int r =
           NativeMethods.   MXSymbolListAtomicSymbolCreators(out num_symbol_creators,  symbol_creators);
            Debug.Assert(r != 0);
            for (int i = 0; i < num_symbol_creators; i++)
            {
                string name = "";
                string description = "";
                uint num_args = 0;
                string[] arg_names = { "" };
                string[] arg_type_infos = { "" };
                string[] arg_descriptions = { "" };
                string key_var_num_args = "";
                string return_type = "";
                r = NativeMethods.MXSymbolGetAtomicSymbolInfo(symbol_creators[i],
                name,
                description,
                num_args,
                arg_names,
                arg_type_infos,
                arg_descriptions,
                key_var_num_args,
                return_type);
                Debug.Assert(r != 0);
                symbol_creators_[name] = symbol_creators[i];
            }
        }

        /*!
        * \brief Get a symbol creator with its name.
        *
        * \param name name of the symbol creator
        * \return handle to the symbol creator
        */
      public  AtomicSymbolCreator GetSymbolCreator(string name)
        {
            return symbol_creators_[name];
        }


        Dictionary<string, AtomicSymbolCreator> symbol_creators_ = new Dictionary<string, AtomicSymbolCreator>();
    }
}
