using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace opwrappergenerator
{
    class EnumType
    {
        private readonly Regex typereg = new Regex("'(.*?)'");
        private readonly Regex typerangereg = new Regex("{(.*)}");
        private string name;
        private string[] enumValues = null;
        public EnumType(string typeName = "ElementWiseOpType", string typeString = "{'avg', 'max', 'sum'}")
        {
            name = typeName;
            if (name.Contains("SoftmaxOutput"))
            {
                
            }
      
            if (typeString.StartsWith("{"))
            {

                var rangematch = typerangereg.Match(typeString);
                if (rangematch.Success)
                {

                    var matchs = typereg.Matches(rangematch.Groups[1].Value);
                    List<string> enums = new List<string>();
                    foreach (Match item in matchs)
                    {
                        enums.Add(item.Groups[1].Value);
                    }
                    enumValues = enums.ToArray();
                }
            }
            else
            {


            }

        }


        public string GetDefinitionString(int indent = 0)
        {
            string ret = "";
            ret += $"public enum {name}\n{{";
            foreach (var value in enumValues)
            {
                if (value == "null")
                {
                    ret += $"_null,\n";
                }
                else
                {
                    ret += $"{value},\n";
                }
            }
            if (enumValues.Length > 0)
            {
                ret = ret.Substring(0, ret.Length - 2);
            }

            ret += "\n};";
            return ret;
        }

        public string GetDefaultValueString(string value = "")
        {
            if (value == "null")
            {
                value = "_null";
            }
            return name + "." + value;
        }
      
    }
}
