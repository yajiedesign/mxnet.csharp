using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using mxnet.numerics.nbase;
using mxnet.numerics.single;
using Razorvine.Pyro;

namespace test.console
{
    class NumericsTest
    {
        public void Test()
        {
            var test = Enumerable.Range(0, 10 * 3 * 4 * 5).Select(s => (float)s).ToArray();
            SingleNArray testsingle = new SingleNArray(new mxnet.numerics.nbase.Shape(10, 3, 4, 5), test);

            var test3_result = testsingle["1::1", "1:3"][":2"];
            var t = test3_result.Flat();

            //Slice[] t1 = new Slice[] {"1:5", "4:6", "0:9"};
            //Slice[] t2 = new Slice[] { "1:2", "1:2", "0:7" ,"3:6"};


            //var z1 = t1.Zip(t2, (l, r) => l?.SubSlice(r) ?? r);

        }
    }
}
