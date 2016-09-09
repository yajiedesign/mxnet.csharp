using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
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
            Regex reg = new Regex("#region Convert(.*?)#endregion", RegexOptions.Singleline);
            var code = File.ReadAllText(@"..\single\SingleNArray.cs");

            var m1 = reg.Match(code);
            var convert = m1.Groups[1].Value;
            code = code.Replace(convert, "#$%^@123");

            code = code.Replace("Single", "Int32");
            code = code.Replace("float", "int");
            code = code.Replace("mxnet.numerics.single", "mxnet.numerics.int32".ToLower());
            code = code.Replace("#region Convert#$%^@123#endregion", $"#region Convert{convert}#endregion");





            IList<Tuple<string, string>> genlist = new List<Tuple<string, string>>()
            {
                Tuple.Create("Int32", "int")
            };

            foreach (var genitem in genlist)
            {
               


            }
        }
    }
}
