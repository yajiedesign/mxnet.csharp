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
        private readonly Regex typereg = new Regex("'(.*?)'");
        private readonly Regex typerangereg = new Regex("{(.*)}");
        private string name;
        private string[] enumValues = null;
        public EnumType(string typeName = "ElementWiseOpType", string typeString = "{'avg', 'max', 'sum'}")
        {
            name = typeName;
      
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


        public string GetDefinitionString()
        {
            string ret = "";
            ret += $"public enum {ToCamelCase(name)}\n{{";
            foreach (var value in enumValues)
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
            if (enumValues.Length > 0)
            {
                ret = ret.Substring(0, ret.Length - 2);
            }

            ret += "\n};";
            return ret;
        }

        private static string ToCamelCase(string name)
        {
            CultureInfo cultureInfo = Thread.CurrentThread.CurrentCulture;
            TextInfo textInfo = cultureInfo.TextInfo;
            return textInfo.ToTitleCase(name).Replace("_","");
        }

        public string GetConvertString()
        {
            string ret = "";
            ret += $"private static readonly List<string> {ToCamelCase(name)}Convert = new List<string>(){{";
            foreach (var value in enumValues)
            {

                    ret += $"\"{value}\",";
                
            }
            if (enumValues.Length > 0)
            {
                ret = ret.Substring(0, ret.Length - 1);
            }

            ret += "};";
            return ret;
        }

        public string GetDefaultValueString(string value = "")
        {
            return ToCamelCase(name) + "." + ToCamelCase(value);
        }

        public string GetName()
        {
            return ToCamelCase(name);
        }
    }
}
