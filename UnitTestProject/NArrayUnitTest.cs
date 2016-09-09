using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using mxnet.numerics.nbase;
using mxnet.numerics.single;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Razorvine.Pyro;
using Razorvine.Serpent;

namespace UnitTestProject
{
    [Serializable]
    public class Parameter<T>
    {
        public Parameter(string key, string shape, T[] value)
        {
            this.Key = key;
            this.Shape = shape;
            this.Value = value;
        }

        public string Key { get; set; }

        public string Shape { get; set; }

        public T[] Value { get; set; }

        public static IDictionary Convert(object obj)
        {
            var o = obj as Parameter<T>;

            Dictionary<string, object> ret = new Dictionary<string, object>();
            ret.Add(nameof(Shape), o.Shape);
            ret.Add(nameof(Value), o.Value);
            return ret;
        }
    }


    [TestClass]
    public class NArrayUnitTest
    {
        private NameServerProxy _ns;
        private PyroProxy _testserver;

        private List<T> eval_array<T>(string code, params Parameter<T>[] param)
        {
            var dict = param.ToDictionary(k => k.Key, v => v);
            object result = _testserver.call("run", "eval_array", code, dict);
            return ((List<object>) result).Select(s => (T) Convert.ChangeType(s, typeof(T))).ToList();
        }

        private T eval_scalar<T>(string code, params Parameter<T>[] param)
        {
            var dict = param.ToDictionary(k => k.Key, v => v);
            object result = _testserver.call("run", "eval_scalar", code, dict);
            return (T) Convert.ChangeType(result, typeof(T));
        }

        [TestInitialize]
        public void Init()
        {
            _ns = NameServerProxy.locateNS(null);
            _testserver = new PyroProxy(_ns.lookup("mxnet.csharp.testserver"));

            Serializer.RegisterClass(typeof(Parameter<float>), Parameter<float>.Convert);
        }

        [TestCleanup]
        public void Cleanup()
        {
            _testserver.Dispose();
            _testserver = null;
            _ns.Dispose();
            _ns = null;
        }


        [TestMethod]
        public void TestSlice1()
        {
            var test = Enumerable.Range(0, 10*4*5*5).Select(s => (float) s).ToArray();
            string testshape = "(10,4,5,5)";
            SingleNArray testsingle = new SingleNArray(new mxnet.numerics.nbase.Shape(10, 4, 5, 5), test);

            List<float> test1NpResult = eval_array<float>("arr[:,::-1,::2,::-2]",
                new Parameter<float>("arr", testshape, test));
            var test1Result = testsingle[":", "::-1", "::2", "::-2"];
            CollectionAssert.AreEqual(test1NpResult, test1Result.Flat().Data, "1");

            List<float> test2NpResult = eval_array<float>("arr[1:9:1,1:3:2,9:1:-1,3::-1]",
                new Parameter<float>("arr", testshape, test));
            var test2Result = testsingle["1:9:1", "1:3:2", "9:1:-1", "3::-1"];
            CollectionAssert.AreEqual(test2NpResult, test2Result.Flat().Data, "2");

            List<float> test3NpResult = eval_array<float>("arr[1:-1,3:1:-1,:]",
                new Parameter<float>("arr", testshape, test));
            var test3Result = testsingle["1:-1", "3:1:-1", ":"];
            CollectionAssert.AreEqual(test3NpResult, test3Result.Flat().Data, "3");

            List<float> test4NpResult = eval_array<float>("arr[-1:-1:3,:1:-1]",
                new Parameter<float>("arr", testshape, test));
            var test4Result = testsingle["-1:-1:3", ":1:-1"];
            CollectionAssert.AreEqual(test4NpResult, test4Result.Flat().Data, "4");

            List<float> test5NpResult = eval_array<float>("arr[-3:-1:3,:1:-1,:10]",
                new Parameter<float>("arr", testshape, test));
            var test5Result = testsingle["-3:-1:3", ":1:-1", ":10"];
            CollectionAssert.AreEqual(test5NpResult, test5Result.Flat().Data, "5");


            List<float> test6NpResult = eval_array<float>("arr[1:5:2,10::-1,2]",
                new Parameter<float>("arr", testshape, test));
            var test6Result = testsingle["1:5:2", "10::-1", "2"];
            CollectionAssert.AreEqual(test6NpResult, test6Result.Flat().Data, "6");


            var test7NpResult = eval_scalar<float>("arr[1:5][3,2,3,2]", new Parameter<float>("arr", testshape, test));
            var test7Result = testsingle["1:5"][3,2,3,2];
            Assert.AreEqual(test7NpResult, test7Result);

            AssertExtension.Throws<ArgumentException>(() => testsingle["::0"].Flat());
            AssertExtension.Throws<ArgumentException>(() => testsingle["xx::0"].Flat());
            AssertExtension.Throws<ArgumentException>(() => testsingle[":xx:0"].Flat());
        }

        [TestMethod]
        public void TestSlice2()
        {
            var test = Enumerable.Range(0, 10*3*4*5).Select(s => (float) s).ToArray();
            string testshape = "(10,3,4,5)";
            SingleNArray testsingle = new SingleNArray(new mxnet.numerics.nbase.Shape(10, 3, 4, 5), test);

            List<float> test1NpResult = eval_array<float>("arr[1::1,1:3]",
                new Parameter<float>("arr", testshape, test));
            var test1Result = testsingle["1::1", "1:3"];
            CollectionAssert.AreEqual(test1NpResult, test1Result.Flat().Data, "1");

            List<float> test2NpResult = eval_array<float>("arr[1::1,1:3][:2,:100,::-1]",
                new Parameter<float>("arr", testshape, test));
            var test2Result = testsingle["1::1", "1:3"][":2", ":100", "::-1"];
            CollectionAssert.AreEqual(test2NpResult, test2Result.Flat().Data, "2");

            List<float> test3NpResult = eval_array<float>("arr[1::1,1:3][1:-1,3:1:-1,:]",
                new Parameter<float>("arr", testshape, test));
            var test3Result = testsingle["1::1", "1:3"]["1:-1", "3:1:-1", ":"];
            CollectionAssert.AreEqual(test3NpResult, test3Result.Flat().Data, "3");

            List<float> test4NpResult = eval_array<float>("arr[::-1,3:1:-1][-3:-1:1,:100:1]",
                new Parameter<float>("arr", testshape, test));
            var test4Result = testsingle["::-1", "3:1:-1"]["-3:-1:1", ":100:1"];
            CollectionAssert.AreEqual(test4NpResult, test4Result.Flat().Data, "4");
            Assert.AreNotEqual(test4Result.Flat().Data.Count(), 0);

            List<float> test5NpResult = eval_array<float>("arr[::-1,3:1:-1][5:1:-1,100::-1,:10]",
                new Parameter<float>("arr", testshape, test));
            var test5Result = testsingle["::-1", "3:1:-1"]["5:1:-1", "100::-1", ":10"];
            CollectionAssert.AreEqual(test5NpResult, test5Result.Flat().Data, "5");
        }

        [TestMethod]
        public void TestSlice3()
        {
            var test = Enumerable.Range(0, 10 * 3 * 4 * 5).Select(s => (float)s).ToArray();
            string testshape = "(10,3,4,5)";
            SingleNArray testsingle = new SingleNArray(new Shape(10, 3, 4, 5), test);

            List<float> test1NpResult = eval_array<float>("arr[[1,3,5,7],[0,2,0,2]]",
                new Parameter<float>("arr", testshape, test));
            var test1Result = testsingle[new int[] {1,3,5,7},new int[] {0,2,0,2}];
            CollectionAssert.AreEqual(test1NpResult, test1Result.Flat().Data, "1");

        }

        [TestMethod]
        public void TestArithmetic()
        {
            Random rnd = new Random(0);
            

            var test = Enumerable.Range(0, 10 * 3 * 4 * 5).Select(s => (float)rnd.Next(0, 50)).ToArray();
            string testshape = "(10,3,4,5)";
            SingleNArray testsingle = new SingleNArray(new Shape(10, 3, 4, 5), test);


            var test1NpResult = eval_scalar<float>("arr[1::1,1:3].sum()", new Parameter<float>("arr", testshape, test));
            var test1Result = testsingle["1::1", "1:3"].Sum();
            Assert.AreEqual(test1NpResult, test1Result, "1");



            var test2NpResult = eval_array<float>("arr.argmax(axis=0)", new Parameter<float>("arr", testshape, test));
            var test2Result = testsingle.Argmax(0);
            CollectionAssert.AreEqual(test2NpResult, test2Result.Flat().Data, "1");

            var a = new float[] {1, 2, 3, 4};
            var b = new float[] {5, 6};
            var aarry = new SingleNArray(new Shape(2, 2),a );
            var barry = new SingleNArray(new Shape(1, 2),b);
            var test3NpResult = eval_array<float>("np.concatenate((a,b),0)", 
                new Parameter<float>("a", "(2,2)", a),
                new Parameter<float>("b", "(1,2)", b));
            var test3Result = SingleNArray.Concatenate(0, aarry, barry);
            CollectionAssert.AreEqual(test3NpResult, test3Result.Flat().Data, "2");

            //var test4_np_result = eval_array<float>("np.concatenate((a,b),axis = 1)",
            //new Parameter<float>("a", "(2,2)", a),
            //new Parameter<float>("b", "(1,2)", b));
            //var test4_result = SingleNArray.Concatenate(1, aarry, barry);
            //CollectionAssert.AreEqual(test4_np_result, test4_result.Flat().data, "2");


        }
    }
}
