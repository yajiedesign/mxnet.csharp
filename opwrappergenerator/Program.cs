using System;
using System.Collections.Generic;
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
            opWrapperGenerator.ParseAllOps();
        }
    }
}
