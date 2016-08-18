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
        private readonly Regex typereg = new Regex("'(.*)'");
        private string name;
        private string[] enumValues = null;
        public EnumType(string typeName = "ElementWiseOpType", string typeString = "{'avg', 'max', 'sum'}")
        {
            name = typeName;
      
            if (typeString.StartsWith("{"))
            {
                var matchs = typereg.Matches(typeString);
                List<string> enums = new List<string>();
                foreach (Match item in matchs)
                {
                    enums.Add(item.Groups[1].Value);
                }
                enumValues = enums.ToArray();
            }
            else
            {


            }

        }


        public string GetDefinitionString(int indent = 0)
        {
            string ret = "";
            ret += $"enum {name}\n{{";
            foreach (var value in enumValues)
            {
                ret += $"{value},\n";
            }
            ret = ret.Substring(0, ret.Length -2);
            ret += "\n};";
            return ret;
        }

        public string GetDefaultValueString(string value = "")
        {
            return name + "." + value;
        }
      
    }
}
