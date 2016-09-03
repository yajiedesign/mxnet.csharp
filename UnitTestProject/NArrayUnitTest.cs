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
        public void TestSlice()
        {
            var rand_num = new Random();
            var test = Enumerable.Range(0, 32 * 3 * 3 * 3).Select(s => (float)s).ToArray();

            List<float> result = eval_array<float>("arr[1:5:2,2:3]", new Parameter<float>("arr", "(32,3,3,3)", test));
            SingleNArray testsingle = new SingleNArray(new mxnet.numerics.nbase.Shape(32, 3, 3, 3), test);
            var result2 = testsingle["1:5:2","2:3"];
            CollectionAssert.AreEqual(result, result2.data);
        }


        private List<T> eval_array<T>(string code, params Parameter<T>[] param)
        {
            var dict = param.ToDictionary(k => k.key, v => v);
            object result = _testserver.call("run", "eval_array", code, dict);
            return ((List<object>)result).Select(s => (T)Convert.ChangeType(s, typeof(T))).ToList();
        }
    }
}
