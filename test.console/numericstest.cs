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
            //var test = Enumerable.Range(0, 10 * 3 * 4 * 5).Select(s => (float)s).ToArray();
            //SingleNArray testsingle = new SingleNArray(new mxnet.numerics.nbase.Shape(10, 3, 4, 5), test);

            //var test5_result = testsingle["::-1", "1:13"]["5:1:-1", "::-1", ":10"];
            //var t1 = test5_result.Flat().data;

            //var t2 = testsingle.Argmax(0);

            Random rnd = new Random(0);


            var test = Enumerable.Range(0, 10 * 3 * 4 * 5).Select(s => (float)rnd.Next(0, 50)).ToArray();
            string testshape = "(10,3,4,5)";
            SingleNArray testsingle = new SingleNArray(new mxnet.numerics.nbase.Shape(10, 3, 4, 5), test);
            var test1_result = testsingle[new int[] { 1, 3, 5, 7 }, new int[] { 0, 2, 0, 2 }];


        }
    }
}
