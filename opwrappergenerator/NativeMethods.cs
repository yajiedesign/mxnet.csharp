using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace opwrappergenerator
{
   static class  NativeMethods
    {        /// Return Type: int
             ///out_size: mx_uint*
             ///out_array: AtomicSymbolCreator**
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolListAtomicSymbolCreators")]
        public static extern int MXSymbolListAtomicSymbolCreators(out uint out_size,
            out IntPtr out_array_ptr);

        /// Return Type: int
        ///creator: AtomicSymbolCreator->void*
        ///name: char**
        ///description: char**
        ///num_args: mx_uint*
        ///arg_names: char***
        ///arg_type_infos: char***
        ///arg_descriptions: char***
        ///key_var_num_args: char**
        ///return_type: char**
        [DllImport("libmxnet.dll", EntryPoint = "MXSymbolGetAtomicSymbolInfo")]
        public static extern int MXSymbolGetAtomicSymbolInfo(IntPtr creator,
          [Out]out IntPtr name,
          [Out]out IntPtr description,
          [Out]out uint num_args,
          [Out]out IntPtr arg_names,
          [Out]out IntPtr arg_type_infos,
          [Out]out IntPtr arg_descriptions,
          [Out]out IntPtr key_var_num_args,
          [Out]out IntPtr return_type);
    }
}
