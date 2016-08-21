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
            uint numSymbolCreators = 0;
            IntPtr symbolCreatorsPtr;

            int r = NativeMethods.MXSymbolListAtomicSymbolCreators(out numSymbolCreators, out symbolCreatorsPtr);
            Debug.Assert(r == 0);
            IntPtr[] symbolCreators = new IntPtr[numSymbolCreators];
            Marshal.Copy(symbolCreatorsPtr, symbolCreators, 0, (int)numSymbolCreators);

            string ret = "";
            for (int i = 0; i < numSymbolCreators; i++)
            {
                IntPtr namePtr;
                IntPtr descriptionPtr;
                uint numArgs;
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
                string name = Marshal.PtrToStringAnsi(namePtr);
                if (name.StartsWith("_"))
                {
                    continue;
                }

                string description = Marshal.PtrToStringAnsi(descriptionPtr);

                IntPtr[] argNamesArray = new IntPtr[numArgs];
                IntPtr[] argTypeInfosArray = new IntPtr[numArgs];
                IntPtr[] argDescriptionsArray = new IntPtr[numArgs];
                if (numArgs > 0)
                {
                    Marshal.Copy(argNamesPtr, argNamesArray, 0, (int) numArgs);
                    Marshal.Copy(argTypeInfosPtr, argTypeInfosArray, 0, (int) numArgs);
                    Marshal.Copy(argDescriptionsPtr, argDescriptionsArray, 0, (int) numArgs);

                }

                List<Arg> args = new List<Arg>();
                for (int j = 0; j < numArgs; j++)
                {
                    var descriptions = Marshal.PtrToStringAnsi(argDescriptionsArray[j]);
                    if (descriptions.Contains("Deprecated"))
                    {
                        continue;
                    }
                    Arg arg = new Arg(name,
                        Marshal.PtrToStringAnsi(argNamesArray[j]),
                        Marshal.PtrToStringAnsi(argTypeInfosArray[j]),
                     descriptions
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
