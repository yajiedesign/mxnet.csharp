using System;
using System.Collections.Generic;
using System.Diagnostics;
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
            var test = Enumerable.Range(0, 100 * 30 * 40 * 50).Select(s => (float)s).ToArray();
            SingleNArray testsingle = new SingleNArray(new mxnet.numerics.nbase.Shape(100, 30, 40, 50), test);

            var test5_result = testsingle["::-1", "1:13"]["5:1:-1", "::-1", ":10"];
            var t1 = test5_result.Flat();
            var t2 = test5_result.Flat2();

            {
                Stopwatch sp = new Stopwatch();
                sp.Start();
                for (int i = 0; i < 10000; i++)
                {
                    var tt1 = test5_result.Flat();
                }
                sp.Stop();
                Console.WriteLine(sp.ElapsedMilliseconds);
            }

            {
                Stopwatch sp = new Stopwatch();
                sp.Start();
                for (int i = 0; i < 10000; i++)
                {
                    var tt1 = test5_result.Flat2();
                }
                sp.Stop();
                Console.WriteLine(sp.ElapsedMilliseconds);
            }
        }
    }
}
