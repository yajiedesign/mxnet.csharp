using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using mxnet.numerics.nbase;

namespace mxnet.numerics.single
{
    public partial struct SingleCalculator : ICalculator<float>
    {
        public float Compare(float a, float b)
        {
            return (Math.Abs(a - b)) < float.Epsilon ? 1 : 0;
        }

    }


    public class TemplateTemp
    {
        public void Template()
        {

            var code = File.ReadAllText(@"..\single\SingleNArray.cs");
            code = code.Replace("Single", "Int32");
            code = code.Replace("float", "int");
            code = code.Replace("mxnet.numerics.single", "mxnet.numerics.int32");



        }
    }
}
