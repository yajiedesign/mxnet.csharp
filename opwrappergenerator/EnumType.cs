using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace opwrappergenerator
{
    class EnumType
    {
        private static readonly Regex TypeReg = new Regex("'(.*?)'");
        private static readonly Regex TypeRangeReg = new Regex("{(.*)}");
        private readonly string[] _enum_values = null;

        public string name { get; }
        public EnumType(string type_name = "ElementWiseOpType", string type_string = "{'avg', 'max', 'sum'}")
        {
            name = ToCamelCase(type_name);

            if (type_string.StartsWith("{"))
            {

                var rangematch = TypeRangeReg.Match(type_string);
                if (rangematch.Success)
                {

                    var matchs = TypeReg.Matches(rangematch.Groups[1].Value);
                    List<string> enums = new List<string>();
                    foreach (Match item in matchs)
                    {
                        enums.Add(item.Groups[1].Value);
                    }
                    _enum_values = enums.ToArray();
                }
            }
        }


        public string GetDefinitionString()
        {
            string ret = "";
            ret += $"public enum {name}\n{{";
            foreach (var value in _enum_values)
            {
                if (value == "null")
                {
                    ret += $"Null,\n";
                }
                else
                {
                    ret += $"{ToCamelCase(value)},\n";
                }
            }
            if (_enum_values.Length > 0)
            {
                ret = ret.Substring(0, ret.Length - 2);
            }

            ret += "\n};";
            return ret;
        }



        public string GetConvertString()
        {
            string ret = "";
            ret += $"private static readonly List<string> {name}Convert = new List<string>(){{";
            foreach (var value in _enum_values)
            {
                ret += $"\"{value}\",";
            }
            if (_enum_values.Length > 0)
            {
                ret = ret.Substring(0, ret.Length - 1);
            }

            ret += "};";
            return ret;
        }

        public string GetDefaultValueString(string value = "")
        {
            return name + "." + ToCamelCase(value);
        }

        private static string ToCamelCase(string name)
        {
            CultureInfo culture_info = Thread.CurrentThread.CurrentCulture;
            TextInfo text_info = culture_info.TextInfo;
            return text_info.ToTitleCase(name).Replace("_", "");
        }
    }
}
