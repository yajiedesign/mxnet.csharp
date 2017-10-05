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

            int r = NativeMethods.MXSymbolListAtomicSymbolCreators(out uint numSymbolCreators, out IntPtr symbolCreatorsPtr);
            Debug.Assert(r == 0);
            IntPtr[] symbolCreators = new IntPtr[numSymbolCreators];
            Marshal.Copy(symbolCreatorsPtr, symbolCreators, 0, (int)numSymbolCreators);

            string retSymbol = "";
            string retNdArray = "";
            string retEnums = "";
            for (int i = 0; i < numSymbolCreators; i++)
            {
                r = NativeMethods.MXSymbolGetAtomicSymbolInfo(symbolCreators[i],
                out IntPtr namePtr,
                out IntPtr descriptionPtr,
                out uint numArgs,
                out IntPtr argNamesPtr,
                out IntPtr argTypeInfosPtr,
                out IntPtr argDescriptionsPtr,
                out IntPtr keyVarNumArgsPtr,
                out IntPtr returnTypePtr);
                string name = Marshal.PtrToStringAnsi(namePtr);
                if (name == null)
                {
                    Console.WriteLine($"error namePtr {i}");
                    continue;;
                }
                if (name.StartsWith("_"))
                {
                    //continue;
                }

                string description = Marshal.PtrToStringAnsi(descriptionPtr);

                IntPtr[] argNamesArray = new IntPtr[numArgs];
                IntPtr[] argTypeInfosArray = new IntPtr[numArgs];
                IntPtr[] argDescriptionsArray = new IntPtr[numArgs];
                if (numArgs > 0)
                {
                    Marshal.Copy(argNamesPtr, argNamesArray, 0, (int)numArgs);
                    Marshal.Copy(argTypeInfosPtr, argTypeInfosArray, 0, (int)numArgs);
                    Marshal.Copy(argDescriptionsPtr, argDescriptionsArray, 0, (int)numArgs);

                }

                List<Arg> args = new List<Arg>();
                for (int j = 0; j < numArgs; j++)
                {
                    string descriptions = Marshal.PtrToStringAnsi(argDescriptionsArray[j]);
                    if (descriptions==null || descriptions.Contains("Deprecated"))
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
