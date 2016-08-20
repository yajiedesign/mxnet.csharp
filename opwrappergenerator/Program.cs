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
           

            string str = @"using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
// ReSharper disable UnusedMember.Global

namespace mxnet.csharp
{
    public partial class OperatorWarp
    {" + opWrapperGenerator.ParseAllOps().Replace("\n","\r\n") + 
    @"}
}
";
            File.WriteAllText(@"..\..\..\mxnet.csharp\OperatorWarp.cs", str);

        }
    }
}
