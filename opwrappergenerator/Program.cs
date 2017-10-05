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
            OpWrapperGenerator opWrapperGenerator = new OpWrapperGenerator();

            var (symbol, ndArray, enums) = opWrapperGenerator.ParseAllOps();

            symbol = symbol.Replace("\n", "\r\n");
            ndArray = ndArray.Replace("\n", "\r\n");
            enums = enums.Replace("\n", "\r\n");

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
" + symbol +
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
" + ndArray +
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
" + enums +  @"
                              
}
";
            File.WriteAllText(@"..\..\..\..\mxnet.csharp\OperatorWarpEnum.cs", strEnum);

        }
    }
}
