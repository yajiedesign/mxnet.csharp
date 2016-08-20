using System;
using System.Linq;
using System.Collections.Generic;
using System.Globalization;
using System.Text.RegularExpressions;
using System.Threading;

namespace opwrappergenerator
{
    internal class Op
    {
        private static readonly Regex R = new Regex(@"
                (?<=[A-Z])(?=[A-Z][a-z]) |
                 (?<=[^A-Z])(?=[A-Z]) |
                 (?<=[A-Za-z])(?=[^A-Za-z])", RegexOptions.IgnorePatternWhitespace| RegexOptions.Compiled);

        private readonly string _name;
        private readonly string _description;
        private readonly List<Arg> _args;

        public Op(string name, string description, List<Arg> args)
        {
            this._name = name;
            this._description = description;

            var nameArg = new Arg(name,
                "symbol_name",
                "string",
                "name of the resulting symbol");
            args.Insert(0, nameArg);
            this._args = args.Where(w => !w.HasDefault).Concat(args.Where(w => w.HasDefault)).ToList();

        }
        /// <summary>
        /// 
        /// </summary>
        enum MyEnum
        { 
            
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="use_name"></param>
        /// <returns></returns>
        public string GetOpDefinitionString(bool use_name)
        {
  
            string ret = "";
            var argsLocal = this._args.Skip(use_name ? 0 : 1).ToList();


            //enum
            if (use_name)
            {
                foreach (var arg in argsLocal.Where(w => w.IsEnum))
                {
                    ret += $"/// <summary>\n/// {arg.Description.Replace("\n","")}\n/// </summary>\n";
                    ret += arg.Enum.GetDefinitionString() +"\n";
                    ret += arg.Enum.GetConvertString() + "\n";
                }


            }


            //comments 
            ret += $"/// <summary>\n/// {_description.Replace("\n", "")}\n/// </summary>\n";


            foreach (var arg in argsLocal)
            {
                ret += $"/// <param name=\"{arg.Name}\">{arg.Description.Replace("\n", "")}</param>\n";
            }
            ret += $" /// <returns>returns new symbol</returns>\n";


            ret += $"public static Symbol {ConvertName(_name)}(";
            foreach (var arg in argsLocal)
            {
                ret += $"{arg.TypeName} {arg.Name}";
                if (arg.HasDefault)
                {

                    ret += $"={arg.DefaultString}";
                }
                ret += ",\n";
            }
            if (argsLocal.Count > 0)
            {
                ret = ret.Substring(0, ret.Length - 2);
            }
      
            ret += ")\n{";

            foreach (var arg in argsLocal)
            {
                ret += arg.DefaultStringWithObject ;
            }

            ret += $"\nreturn new Operator(\"{_name}\")\n";

            foreach (var arg in _args)
            {
                if (arg.TypeName == "Symbol" ||
                    arg.TypeName == "Symbol[]" ||
                    arg.Name == "symbolName")
                {
                    continue;
                }

                if (arg.IsEnum)
                {
                    ret += $".SetParam(\"{arg.OrginName}\", {arg.Enum.Name}Convert[(int){arg.Name}])\n";
                }
                else
                {
                    ret += $".SetParam(\"{arg.OrginName}\", {arg.Name})\n";
                }
        

            }


            foreach (var arg in _args)
            {
                if (arg.TypeName != "Symbol")
                {
                    continue;
                }
                ret += $".SetInput(\"{arg.OrginName}\", {arg.Name})\n";
            }

            foreach (var arg in _args)
            {
                if (arg.TypeName != "Symbol[]")
                {
                    continue;
                }
                ret += $".AddInput({arg.Name})\n";
            }
            if (use_name)
            {
                ret += ".CreateSymbol(symbolName);\n";
            }
            else
            {

                ret += ".CreateSymbol();\n";
            }
            ret += "}";
            return ret;
        }

        private string ConvertName(string name)
        {
            CultureInfo cultureInfo = Thread.CurrentThread.CurrentCulture;
            TextInfo textInfo = cultureInfo.TextInfo;



            var ret = R.Replace(name, "_");
            return textInfo.ToTitleCase(ret).Replace("_", "");
        }
    }
}