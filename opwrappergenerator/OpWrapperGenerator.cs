using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace opwrappergenerator
{
    class OpWrapperGenerator
    {
        public (string, string, string) ParseAllOps()
        {
            uint num_symbol_creators = 0;
            IntPtr symbol_creators_ptr;

            int r = NativeMethods.MXSymbolListAtomicSymbolCreators(out num_symbol_creators, out symbol_creators_ptr);
            Debug.Assert(r == 0);
            IntPtr[] symbol_creators = new IntPtr[num_symbol_creators];
            Marshal.Copy(symbol_creators_ptr, symbol_creators, 0, (int)num_symbol_creators);

            string retSymbol = "";
            string retNdArray = "";
            string retEnums = "";
            for (int i = 0; i < num_symbol_creators; i++)
            {
                IntPtr name_ptr;
                IntPtr description_ptr;
                uint num_args;
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
                string name = Marshal.PtrToStringAnsi(name_ptr);
                if (name.StartsWith("_"))
                {
                    continue;
                }

                string description = Marshal.PtrToStringAnsi(description_ptr);

                IntPtr[] arg_names_array = new IntPtr[num_args];
                IntPtr[] arg_type_infos_array = new IntPtr[num_args];
                IntPtr[] arg_descriptions_array = new IntPtr[num_args];
                if (num_args > 0)
                {
                    Marshal.Copy(arg_names_ptr, arg_names_array, 0, (int)num_args);
                    Marshal.Copy(arg_type_infos_ptr, arg_type_infos_array, 0, (int)num_args);
                    Marshal.Copy(arg_descriptions_ptr, arg_descriptions_array, 0, (int)num_args);

                }

                List<Arg> args = new List<Arg>();
                for (int j = 0; j < num_args; j++)
                {
                    var descriptions = Marshal.PtrToStringAnsi(arg_descriptions_array[j]);
                    if (descriptions.Contains("Deprecated"))
                    {
                        continue;
                    }
                    Arg arg = new Arg(name,
                        Marshal.PtrToStringAnsi(arg_names_array[j]),
                        Marshal.PtrToStringAnsi(arg_type_infos_array[j]),
                     descriptions
                        );
                    args.Add(arg);
                }
                string tmp = "";
                var op = new Op(name, description, args);
                retSymbol += op.GetOpDefinitionString(true, OpNdArrayOrSymbol.Symbol, ref retEnums) + "\n"
                     + op.GetOpDefinitionString(false, OpNdArrayOrSymbol.Symbol,ref tmp) + "\n";


                retNdArray += op.GetOpDefinitionString(true, OpNdArrayOrSymbol.NdArray, ref tmp) + "\n"
                     +  op.GetOpDefinitionString(false, OpNdArrayOrSymbol.NdArray, ref tmp) + "\n";

            }
            return (retSymbol, retNdArray, retEnums);
        }
    }
}
