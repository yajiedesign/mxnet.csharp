using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using mxnet.numerics.single;
using Razorvine.Pyro;

namespace test.console
{
    class NumericsTest
    {
        public void Test()
        {

            using (NameServerProxy ns = NameServerProxy.locateNS(null))
            {
                var l = ns.list(null, null);
                using (PyroProxy something = new PyroProxy(ns.lookup("mxnet.csharp.testserver")))
                {




                }
            }


            //test Slice
            SingleNArray testsingle = new SingleNArray(new mxnet.numerics.nbase.Shape(32, 3, 20, 60));
            var t2 = testsingle["2:5:2", ":3"];

        }
    }
}
