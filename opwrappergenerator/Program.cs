using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace opwrappergenerator
{
    class Program
    {
        static void Main(string[] args)
        {
            OpWrapperGenerator op_wrapper_generator = new OpWrapperGenerator();

            var (Symbol, NdArray, Enums) = op_wrapper_generator.ParseAllOps();

            Symbol = Symbol.Replace("\n", "\r\n");
            NdArray = NdArray.Replace("\n", "\r\n");
            Enums = Enums.Replace("\n", "\r\n");

            string strSymbol = @"using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
// ReSharper disable UnusedMember.Global

namespace mxnet.csharp
{
    public partial class Symbol
    {
" + Symbol +
    @"}
}
";
            File.WriteAllText(@"..\..\..\..\mxnet.csharp\OperatorWarpSymbol.cs", strSymbol);


            string strNdArray = @"using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
// ReSharper disable UnusedMember.Global

namespace mxnet.csharp
{
    public partial class NdArray
    {
" + NdArray +
                               @"}
}
";
            File.WriteAllText(@"..\..\..\..\mxnet.csharp\OperatorWarpNdArray.cs", strNdArray);


            string strEnum = @"using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
// ReSharper disable UnusedMember.Global

namespace mxnet.csharp
{
" + Enums +  @"
                              
}
";
            File.WriteAllText(@"..\..\..\..\mxnet.csharp\OperatorWarpEnum.cs", strEnum);

        }
    }
}
