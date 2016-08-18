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
        public string ParseAllOps()
        {
            uint num_symbol_creators = 0;
            IntPtr symbol_creators_ptr = IntPtr.Zero;

            int r = NativeMethods.MXSymbolListAtomicSymbolCreators(out num_symbol_creators, out symbol_creators_ptr);
            Debug.Assert(r == 0);
            IntPtr[] symbol_creators = new IntPtr[num_symbol_creators];
            Marshal.Copy(symbol_creators_ptr, symbol_creators, 0, (int)num_symbol_creators);

            string ret = "";
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
                string name = Marshal.PtrToStringAnsi(name_ptr);
                string description = Marshal.PtrToStringAnsi(description_ptr);

                IntPtr[] arg_names_array = new IntPtr[num_args];
                IntPtr[] arg_type_infos_array = new IntPtr[num_args];
                IntPtr[] arg_descriptions_array = new IntPtr[num_args];
                if (num_args > 0)
                {
                    Marshal.Copy(arg_names_ptr, arg_names_array, 0, (int) num_args);
                    Marshal.Copy(arg_type_infos_ptr, arg_type_infos_array, 0, (int) num_args);
                    Marshal.Copy(arg_descriptions_ptr, arg_descriptions_array, 0, (int) num_args);

                }

                List<Arg> args = new List<Arg>();
                for (int j = 0; j < num_args; j++)
                {
                    Arg arg = new Arg(name,
                        Marshal.PtrToStringAnsi(arg_names_array[j]),
                        Marshal.PtrToStringAnsi(arg_type_infos_array[j]),
                        Marshal.PtrToStringAnsi(arg_descriptions_array[j])
                        );
                    args.Add(arg);
                }

                var op = new Op(name, description, args);
                ret += op.GetOpDefinitionString(true) + "\n"
                           + op.GetOpDefinitionString(false) + "\n";


            }
            return ret;
        }
    }
}
