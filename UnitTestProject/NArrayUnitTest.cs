using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using mxnet.numerics.single;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Razorvine.Pyro;
using Razorvine.Serpent;

namespace UnitTestProject
{
    [Serializable]
    public class Parameter<T>
    {
        public Parameter(string key, string shape ,T[] value)
        {
            this.key = key;
            this.shape = shape;
            this.value = value;
        }

        public string key { get; set; }

        public string shape { get; set; }

        public T[] value { get; set; }

        public static IDictionary Convert(object obj)
        {
            var o = obj as Parameter<T>;

            Dictionary<string, object> ret = new Dictionary<string, object>();
            ret.Add(nameof(shape), o.shape);
            ret.Add(nameof(value), o.value);
            return ret;
        }
    }


    [TestClass]
    public class NArrayUnitTest
    {
        private NameServerProxy _ns;
        private PyroProxy _testserver;

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
            var test = Enumerable.Range(0, 10 * 4 * 5 * 5).Select(s => (float)s).ToArray();
            string testshape = "(10,4,5,5)";
            SingleNArray testsingle = new SingleNArray(new mxnet.numerics.nbase.Shape(10, 4, 5, 5), test);

            List<float> test1_np_result = eval_array<float>("arr[:,::-1,::2,::-2]", new Parameter<float>("arr", testshape, test));
            var test1_result = testsingle[":", "::-1", "::2", "::-2"];
            CollectionAssert.AreEqual(test1_np_result, test1_result.Flat().data, "1");

            List<float> test2_np_result = eval_array<float>("arr[1:9:1,1:3:2,9:1:-1,3::-1]", new Parameter<float>("arr", testshape, test));
            var test2_result = testsingle["1:9:1", "1:3:2", "9:1:-1", "3::-1"];
            CollectionAssert.AreEqual(test2_np_result, test2_result.Flat().data, "2");

            List<float> test3_np_result = eval_array<float>("arr[1:-1,3:1:-1,:]", new Parameter<float>("arr", testshape, test));
            var test3_result = testsingle["1:-1", "3:1:-1",":"];
            CollectionAssert.AreEqual(test3_np_result, test3_result.Flat().data, "3");

            List<float> test4_np_result = eval_array<float>("arr[-1:-1:3,:1:-1]", new Parameter<float>("arr", testshape, test));
            var test4_result = testsingle["-1:-1:3", ":1:-1"];
            CollectionAssert.AreEqual(test4_np_result, test4_result.Flat().data, "4");

            List<float> test5_np_result = eval_array<float>("arr[-3:-1:3,:1:-1,:10]", new Parameter<float>("arr", testshape, test));
            var test5_result = testsingle["-3:-1:3", ":1:-1",":10"];
            CollectionAssert.AreEqual(test5_np_result, test5_result.Flat().data, "5");


            List<float> test6_np_result = eval_array<float>("arr[1:5:2,10::-1,2]", new Parameter<float>("arr", testshape, test));
            var test6_result = testsingle["1:5:2", "10::-1","2"];
            CollectionAssert.AreEqual(test6_np_result, test6_result.Flat().data, "6");
        }

        [TestMethod]
        public void TestSlice2()
        {
            var test = Enumerable.Range(0, 10 * 3 * 4 * 5).Select(s => (float)s).ToArray();
            string testshape = "(10,3,4,5)";
            SingleNArray testsingle = new SingleNArray(new mxnet.numerics.nbase.Shape(10, 3, 4, 5), test);

            List<float> test1_np_result = eval_array<float>("arr[1::1,1:3]", new Parameter<float>("arr", testshape, test));
            var test1_result = testsingle["1::1", "1:3"];
            CollectionAssert.AreEqual(test1_np_result, test1_result.Flat().data, "1");

            List<float> test2_np_result = eval_array<float>("arr[1::1,1:3][:2]", new Parameter<float>("arr", testshape, test));
            var test2_result = testsingle["1::1", "1:3"][":2"];
            CollectionAssert.AreEqual(test2_np_result, test2_result.Flat().data, "2");

            List<float> test3_np_result = eval_array<float>("arr[1::1,1:3][1:-1,3:1:-1,:]", new Parameter<float>("arr", testshape, test));
            var test3_result = testsingle["1::1", "1:3"]["1:-1", "3:1:-1", ":"];
            CollectionAssert.AreEqual(test3_np_result, test3_result.Flat().data, "3");

            List<float> test4_np_result = eval_array<float>("arr[1::1,1:3][-1:-1:3,:1:-1]", new Parameter<float>("arr", testshape, test));
            var test4_result = testsingle["1::1", "1:3"]["-1:-1:3", ":1:-1"];
            CollectionAssert.AreEqual(test4_np_result, test4_result.Flat().data, "4");

            List<float> test5_np_result = eval_array<float>("arr[1::1,1:3][-3:-1:3,:1:-1,:10]", new Parameter<float>("arr", testshape, test));
            var test5_result = testsingle["1::1", "1:3"]["-3:-1:3", ":1:-1", ":10"];
            CollectionAssert.AreEqual(test5_np_result, test5_result.Flat().data, "5");
        }

        private List<T> eval_array<T>(string code, params Parameter<T>[] param)
        {
            var dict = param.ToDictionary(k => k.key, v => v);
            object result = _testserver.call("run", "eval_array", code, dict);
            return ((List<object>)result).Select(s => (T)Convert.ChangeType(s, typeof(T))).ToList();
        }
    }
}
